from typing import List, Literal, Union, Tuple
import json
import autode as ade
import matplotlib.pyplot as plt
import numpy as np
import os
from copy import deepcopy
from tqdm import tqdm
import fnmatch
import yaml
from rdkit import Chem
from autode.input_output import xyz_file_to_atoms

from plot.plot_energy_surface import plot_1d_thermo, plot_2d, Smooth, get_xy_value
from kinetics.calculator import tst_kinetics
from kinetics.FitKinParam import fit_kin_param
from rdkit_util import get_decomp_idx, get_changed_bonds
from rdkit_util import str_to_mol, drop_map_num, CalculateSpinMultiplicity
from rdkit_util import is_mapped
from qm.qm_calc_keywords import functional_dict, base_set_dict, dispersion_dict, \
    func_str_dict, base_set_str_dict, dispersion_str_dict, def2svp
from qm.mol_manipulation import translation
from load_write import load_coor_from_xyz_file
from qm.ts_location_tools import initialize_autode_complex, match_atom_nums, translate_rotate_reactant, \
    bond_rearrangement
from qm.thermo import ThermoMol

from exception import AtomMappingException


class TsSearchBase:
    """
    A class for generating the python file for reaction rate calculation
    """

    def __init__(self,
                 output_file_name: str,
                 reactants: dict,
                 products: dict,
                 dir_name: str = None,
                 hmethod: str = 'orca',
                 lmethod: str = 'xtb'):
        """
        Args:
            reactants: reactants {name: smiles}
            products: products {name: smiles}
            dir_name: saved dir

        """
        self.opt_ts_keys_str = None
        self.reac_prod_results = None
        self.rate_constants = None
        self.sw_cfg = {}
        self.reactants = reactants
        self.products = products
        self.output_file_name = output_file_name
        self.cal_free_energy = True
        self.cal_enthalpy = True
        self.temps = None
        self.lfm_method = 'grimme'
        self.lotc = {}  # level of theory for computation
        self.freq_scale_factor = {}
        self.n_cores = 8
        self.max_core = 4000
        self.lmethod = lmethod
        self.hmethod = hmethod
        rsmi, psmi = '.'.join(list(reactants.values())), '.'.join(list(products.values()))
        self.smarts = '>>'.join([Chem.MolToSmiles(Chem.MolFromSmiles(rsmi)),
                                 Chem.MolToSmiles(Chem.MolFromSmiles(psmi))])
        self.rxn_results = None
        # check reactant and product name
        for key in self.reactants.keys():
            if key in self.products.keys():
                raise AttributeError("Same name in both reactants and products")
        if dir_name is None:
            self.dir_name = self.smarts.replace('.', '&').replace('>>', '_to_')
        else:
            self.dir_name = dir_name

        self.initial_path = os.path.join(os.getcwd(), self.dir_name)
        if not os.path.exists(self.initial_path):
            os.makedirs(self.initial_path)
        os.chdir(self.initial_path)

        self.settings, self.settings_str = dict({}), dict({})

        self.settings.update(functional_dict)
        self.settings.update(base_set_dict)
        self.settings.update(dispersion_dict)

        self.settings_str.update(func_str_dict)
        self.settings_str.update(base_set_str_dict)
        self.settings_str.update(dispersion_str_dict)

    def sw_config(self, name: str, n_cores: int = 8, max_core: float = 4000):
        """
        Which software will be used to perform the low and high level calculation
        Args:
            name: name of electronic package
            n_cores: number of cores used for calculation
            max_core: Max memory of every core
        Returns: None
        """
        if name in [self.lmethod, self.hmethod]:
            self.sw_cfg[name] = {"n_cores": n_cores, "max_core": max_core}

    def thermo_config(self,
                      temps: List[float],
                      lfm_method: Literal['igm', 'truhlar', 'grimme', 'minenkov'] = 'grimme',
                      cal_free_energy: bool = True,
                      cal_enthalpy: bool = True):
        """

        Args:
            temps:
            lfm_method: Method to treat low frequency modes. {'igm', 'truhlar', 'grimme'}
            cal_free_energy:
            cal_enthalpy:

        Returns: None
        """
        self.temps = [float(temp) for temp in temps]
        self.cal_enthalpy = cal_enthalpy
        self.cal_free_energy = cal_free_energy
        self.lfm_method = lfm_method

    def level_of_theory_config(self,
                               calc_type: Literal['sp', 'opt', 'opt_ts', 'hess'],
                               sw_name: Literal['orca', 'xtb', 'g16'] = 'orca',
                               functional: Literal['b3lyp', 'm062x', 'wb97xd', 'wb97xd3', 'wb97xd3bj', 'wb97mv',
                               'pwpb95', 'pw6b95', 'pw6b95d3', 'ccsdt'] = 'b3lyp',
                               basis_set: Literal[
                                   'def2tsvp', 'def2tzvp_f', 'def2tzvp', 'def2tzvpp', 'def2qzvp', 'def2qzvpp',
                                   'madef2svp', 'madef2tzvp_f', 'madef2tzvp', 'madef2tzvpp', 'madef2qzvp',
                                   'madef2qzvpp', 'ccpvdz', 'ccpvtz', 'ccpvqz', 'aug_ccpvdz', 'aug_ccpvtz',
                                   'aug_ccpvqz', '631gx', '631+gx', '6311gxx', '6311+gxx', 'def2svp'] = 'def2tzvp',
                               dispersion: Literal['d3bj', 'd3', 'no_dispersion'] = 'd3bj'):
        """
        This function is to set the calculations for optimization, single point calculation
        There are various settings in autodE: https://duartegroup.github.io/autodE/reference/wrappers/keywords.html
        However, we just use the list settings for optimization and sing point energy calculation
        Args:
            sw_name: the name of software
            calc_type: the type of calculation 'sp', 'opt'
            functional: the functional for example, 'B3LYP'
            basis_set: Get the functional in this set of keywords
            dispersion:
        Returns: None

        """
        if sw_name is not None:
            if sw_name not in self.lotc.keys():
                self.lotc[sw_name] = {calc_type: {
                    "functional": functional,
                    "basis_set": basis_set,
                    "dispersion": dispersion}}
            elif calc_type not in self.lotc[sw_name].keys():
                self.lotc[sw_name][calc_type] = {"functional": functional,
                                                 "basis_set": basis_set,
                                                 "dispersion": dispersion}
            else:
                raise KeyError('the key already exists')

    def opt_ts_keys(self,
                    keys: str = "Opt=(TS, NoEigenTest, MaxStep=5, NoTrustUpdate, CalcAll)"):
        self.opt_ts_keys_str = keys

    def _set_ade(self):
        """
        Set autode settings
        Returns: an autode class

        """
        ade.Config.hcode = self.hmethod
        ade.Config.lcode = self.lmethod
        ade.Config.lfm_method = self.lfm_method
        sw_config = getattr(ade.Config, self.hmethod.upper())

        if self.opt_ts_keys_str is not None:
            for i, keywords in enumerate(sw_config.keywords.opt_ts._list):
                if type(keywords).__name__ == 'str':
                    if 'opt' in keywords.lower():
                        sw_config.keywords.opt_ts._list[i] = self.opt_ts_keys_str
                        break

        # Setting cores and memory
        cores, mems = [], []
        for key in [self.lmethod, self.hmethod]:
            if key not in self.sw_cfg.keys():
                raise KeyError(f'Can not find n_cores settings of {key}')
            else:
                cores.append(self.sw_cfg[key]['n_cores'])
                mems.append(self.sw_cfg[key]['max_core'])
        ade.Config.n_cores = max(cores)
        ade.Config.max_core = max(mems)

        # Setting level of theory of software
        for sw, settings in self.lotc.items():
            for calc_type, key_dic in settings.items():
                sw_kwds = getattr(ade.Config, str.upper(sw))
                sw_kwds_set = getattr(sw_kwds.keywords, calc_type)
                for key, sets in key_dic.items():
                    if sets is not None:
                        setattr(sw_kwds_set, key, self.settings[sets])

        return ade

    def load_results(self, path: str):
        """
        Load calculation results
        Args:
            path:

        Returns:

        """
        with open(path, 'r') as f:
            results = json.load(f)
        f.close()
        self.rxn_results = results
        self.temps = [float(temp) for temp in self.rxn_results['G'].keys()]
        self.smarts = self.rxn_results['smarts']

    def reacs_prods_thermo(self, autode_obj, calc_lmethod, calc_hmethod, hmethod_cores):
        """
        Calculating reactant and product thermochemistry
        Note that TsSearch has already calculated thermo of reactants, products,
            and products by method calculate_reaction_profile
        Returns:
        """

        def get_energies(ade_obj, smis: List[str], smi_name: List[str], temperatures):
            thermo_mols = []
            molecules = []
            for i, smi in enumerate(smis):
                thermo_mol = ThermoMol(name=smi)
                molecule = ade_obj.Molecule(name=smi_name[i], smiles=smi)
                molecule.mult = CalculateSpinMultiplicity(smi)
                try:
                    molecule.find_lowest_energy_conformer(lmethod=calc_lmethod)
                except:
                    pass
                molecule.optimise(method=calc_hmethod, n_cores=hmethod_cores)
                molecule.calc_thermo(method=calc_hmethod, n_cores=hmethod_cores)
                molecule.single_point(method=calc_hmethod, n_cores=hmethod_cores)
                thermo_mol.sp = float(molecule.energy.to("kJ mol-1"))
                thermo_mol.zpe = float(molecule.zpe.to("kJ mol-1"))
                for temperature in temperatures:
                    molecule.calc_thermo(method=calc_hmethod, temp=temperature, n_cores=hmethod_cores)
                    thermo_mol.thermo_point(temperature,
                                            g_cont=float(molecule.g_cont.to("kJ mol-1")),
                                            h_cont=float(molecule.h_cont.to("kJ mol-1")))
                molecules.append(molecule)
                thermo_mols.append(thermo_mol)
            return thermo_mols, molecules

        self.reac_prod_results = {}

        # reactant thermo
        print('Calculating reactant thermochemistry...')
        reac_thermo_mols, reactants = get_energies(ade_obj=autode_obj,
                                                   smi_name=list(self.reactants.keys()),
                                                   smis=list(self.reactants.values()),
                                                   temperatures=self.temps)
        self.reac_prod_results['reactants'] = reac_thermo_mols

        # products thermo
        print('Calculating product thermochemistry...')
        prods_thermo_mols, products = get_energies(ade_obj=autode_obj,
                                                   smi_name=list(self.products.keys()),
                                                   smis=list(self.products.values()),
                                                   temperatures=self.temps)
        self.reac_prod_results['products'] = prods_thermo_mols

        return reactants, products

    def _tst_calculation(self,
                         free_energy: List[float],
                         ifreq: List[float],
                         smarts: Union[str, int] = None,
                         temps: List[float] = None,
                         E0_reac: float = None,
                         E0_TS: List[float] = None,
                         E0_prod: float = None
                         ):
        """
        Calculating rate constants using TST theory
        Args:
            free_energy: free energy of calculated reaction points
            ifreq: imaginary frequency of calculated reaction points
            smarts: reaction smarts (only one reaction per class)
            temps: reaction temperatures
            E0_reac: reactant electronic energy
            E0_TS: transition state electronic energy (May different at VTST surface)
            E0_prod: product electronic energy

        Returns: calculated reaction rates

        """
        if smarts is not None:
            if type(smarts).__name__ == 'str':
                sma = len(smarts.split('>>')[0].split('.'))
            else:
                sma = smarts
        else:
            sma = len(self.smarts.split('>>')[0].split('.'))

        temperatures = self.temps if temps is None else temps
        assert len(temperatures) == len(free_energy), "The length of temperature list should equal to that" \
                                                      " of free energy list"

        if len(ifreq) > 1:
            assert len(free_energy) == len(ifreq), "The length of imaginary frequency list should equal to that" \
                                                   " of free energy list or equal to 1"
            checked_ifreq = ifreq
        else:
            checked_ifreq = ifreq * len(free_energy)

        if E0_TS is not None:
            if len(E0_TS) > 1:
                assert len(free_energy) == len(E0_TS), "The length of transition state electronic energy should " \
                                                       "equal to that of imaginary frequency list or equal to 1"
                checked_e0_ts = E0_TS
            else:
                checked_e0_ts = E0_TS * len(free_energy)
        else:
            checked_e0_ts = [None] * len(free_energy)

        # temps = [temp if type(temp).__name__ == 'tuple' else (temp, 'K') for temp in temps]
        try:
            rate_cons = [tst_kinetics(sma, g, temp, i, E0_reac=E0_reac, E0_prod=E0_prod, E0_TS=ts) for
                         g, i, temp, ts in zip(free_energy, checked_ifreq, temps, checked_e0_ts)]
        except:
            rate_cons = [tst_kinetics(sma, g, temp, i) for
                         g, i, temp in zip(free_energy, checked_ifreq, temps)]
            print(Warning("Estimating reaction rate based on the Wigner formula"))

        return rate_cons

    def calc_rate_constants(self, calc_reverse: bool = True):

        """
        Rate constants calculation at default temperature
        To calculate rate constants, reaction information should be calculated firstly.
        That is, the attribute rxn_results should not be None
        results_dic: results dict contain reaction information
                        Containing {'G': {temperature: {'forward': float, 'reverse': float}},
                                    'H': {temperature: {'forward': float, 'reverse': float}},
                                    'E0_reacs': List[float],
                                    'E0_prods': List[float],
                                    'E0_TS': Union[float, List[float],
                                    'ifreq': float}
                        Note: {'G': {'forward': float}} is required at least
        Returns:

        """
        assert hasattr(self, 'rxn_results') and self.rxn_results is not None, "Couldn't get reaction information"
        e0_reacs = self.rxn_results['E0_reacs'] if 'E0_reacs' in self.rxn_results.keys() else None
        e0_prods = self.rxn_results['E0_prods'] if 'E0_prods' in self.rxn_results.keys() else None
        if 'E0_TS' in self.rxn_results.keys():
            if type(self.rxn_results['E0_TS']).__name__ == 'list':
                e0_ts = [e for e in self.rxn_results['E0_TS']]
            elif type(self.rxn_results['E0_TS']).__name__ == 'float':
                e0_ts = [self.rxn_results['E0_TS']]
            else:
                raise TypeError('Unsupported reaction information')
        else:
            e0_ts = None
        ifreq = [float(i) for i in self.rxn_results['ifreq']] if type(self.rxn_results['ifreq']).__name__ == 'list' \
            else [float(self.rxn_results['ifreq'])]
        g_f = [float(self.rxn_results['G'][str(temp)]['forward']) for temp in self.temps]
        rate_constants = self._tst_calculation(free_energy=g_f, ifreq=ifreq, smarts=self.smarts, temps=self.temps,
                                               E0_reac=e0_reacs, E0_TS=e0_ts, E0_prod=e0_prods)
        if calc_reverse:
            g_r = [float(self.rxn_results['G'][str(temp)]['reverse']) for temp in self.temps]
            rate_constants_r = self._tst_calculation(free_energy=g_r, ifreq=ifreq, smarts=self.smarts, temps=self.temps,
                                                     E0_reac=e0_prods, E0_TS=e0_ts, E0_prod=e0_reacs)
        else:
            rate_constants_r = None

        self.rate_constants = {'forward': rate_constants, 'reverse': rate_constants_r}

        return self.rate_constants

    def fit_by_arrhenius(self,
                         results_dic: dict = None,
                         path: str = None,
                         save_fig: str = None,
                         calc_reverse: bool = True,
                         grid: int = 100,
                         save_params: bool = True):
        """
        Fit temperature dependent arrhenius equation
        Args:
            results_dic: results dict contain reaction information
                        Containing {'G': {temperature: {'forward': float, 'reverse': float}},
                                    'H': {temperature: {'forward': float, 'reverse': float}},
                                    'E0_reacs': List[float],
                                    'E0_prods': List[float],
                                    'E0_TS': Union[float, List[float],
                                    'ifreq': float}
                        Note: {'G': {'forward': float}} is required at least

            path: path of results
            save_fig: whether save fitted results
            calc_reverse: whether calculate reverse rate constants
            grid: number of points for drawing fitted kinetics curve
            save_params: save fitted kinetic parameters
        Returns: fitted parameters A, n, Ea

        """
        if results_dic is not None:
            self.rxn_results = results_dic
        elif path is not None:
            self.load_results(path)
        else:
            assert hasattr(self, 'rxn_results'), "Couldn't get reaction information"

        rate_constants = self.calc_rate_constants(calc_reverse=calc_reverse)

        def fit_param(temps: np.array, rate_cons: np.array, calc_temps: np.array):
            kin_params, confidence = fit_kin_param(temps, rate_cons)
            test_rate_cons = np.log(kin_params[0]) + kin_params[1] * np.log(calc_temps) - kin_params[
                2] / 8.314 / calc_temps

            return test_rate_cons, kin_params

        calc_at_temps = np.arange(self.temps[0], self.temps[-1], grid)
        test_cons, params = fit_param(temps=self.temps,
                                      rate_cons=np.array(rate_constants['forward']),
                                      calc_temps=calc_at_temps)
        params_dic = {'forward': {'A': float(params[0]), 'n': float(params[1]), 'Ea': float(params[2])}}
        if save_fig is not None:
            fig = plt.figure()
        else:
            fig = None
        if calc_reverse:
            test_cons_r, params_r = fit_param(temps=np.array(self.temps),
                                              rate_cons=np.array(rate_constants['reverse']),
                                              calc_temps=calc_at_temps)
            params_dic['reverse'] = {'A': float(params_r[0]), 'n': float(params_r[1]), 'Ea': float(params_r[2])}
            if save_fig is not None:
                plt.scatter(1000 / np.array(self.temps), np.log(rate_constants['reverse']), c='#c82423',
                            label='TST calculated $k$ (reverse)')
                plt.plot(1000 / calc_at_temps, test_cons_r, c='#c82423', label='Fitted $k$ (reverse)')
        if save_fig is not None:
            plt.scatter(1000 / np.array(self.temps), np.log(rate_constants['forward']), c='#2878b5',
                        label='TST calculated $k$ (forward)')
            plt.plot(1000 / calc_at_temps, test_cons, c='#2878b5', label='Fitted $k$ (forward)')
            plt.legend()
            plt.xlabel('1000 / Temperature (1000/K)')
            plt.ylabel('$ln k$')
            plt.savefig(save_fig)
            plt.close(fig=fig)
        if save_params:
            with open('kin_parameters.yaml', 'w') as f:
                yaml.dump(params_dic, f)
            f.close()

        return params_dic

    @property
    def avila_func(self):
        """return keywords of available functional"""
        return list(functional_dict.keys())

    @property
    def avila_base_set(self):
        """return keywords of available base-set"""
        return list(base_set_dict.keys())

    @property
    def avila_dispersion(self):
        """return keywords of available base-set"""
        return list(dispersion_dict.keys())


class TsSearch(TsSearchBase):
    """
    A class for transition state calculation
    This object employ the autodE obj directly
    """

    def __init__(self,
                 output_file_name: str,
                 reactants: dict,
                 products: dict,
                 dir_name: str = None,
                 lmethod: str = 'xtb',
                 hmethod: str = 'orca'):
        super().__init__(output_file_name=output_file_name,
                         reactants=reactants,
                         products=products,
                         dir_name=dir_name,
                         lmethod=lmethod,
                         hmethod=hmethod)

    def ts_search(self):
        """
        For standard reaction profile generation process
        single point energy and freq tasks were automatically performed
        Thus, we just extract reaction information from rxn object

        Returns:

        """
        ade_obj = self._set_ade()
        species = []
        for key, val in self.reactants.items():
            species.append(ade_obj.Reactant(smiles=val))
        for key, val in self.products.items():
            species.append(ade_obj.Product(smiles=val))

        rxn = ade_obj.Reaction(*species, name=self.dir_name.replace('.', '&').replace('>>', '_to_'))
        # rxn.calculate_reaction_profile(free_energy=True)
        rxn.calculate_reaction_profile(enthalpy=True)

        ifreq = float(rxn.ts.imaginary_frequencies[0])
        e0_reacs = []
        for mol in rxn.reacs:
            e0_reacs.append(float(mol.energy.to('kJ mol-1') + mol.zpe.to('kJ mol-1')))
        e0_prods = []
        for mol in rxn.prods:
            e0_prods.append(float(mol.energy.to('kJ mol-1') + mol.zpe.to('kJ mol-1')))
        e0_ts = float(rxn.ts.energy.to('kJ mol-1') + rxn.ts.zpe.to('kJ mol-1'))

        results = {'G': {}, 'H': {}, 'ifreq': ifreq, 'E0_reacs': float(np.sum(e0_reacs)),
                   'E0_prods': float(np.sum(e0_prods)), 'E0_TS': e0_ts}

        for temp in self.temps:
            rxn.temp = temp
            rxn.calculate_thermochemical_cont()
            g_reacs, h_reacs = 0, 0
            for mol in rxn.reacs:
                g_reacs += mol.free_energy.to('kJ mol-1')
                h_reacs += mol.enthalpy.to('kJ mol-1')
            g_prods, h_prods = 0, 0
            for mol in rxn.prods:
                g_prods += mol.free_energy.to('kJ mol-1')
                h_prods += mol.enthalpy.to('kJ mol-1')
            g_ts, h_ts = rxn.ts.free_energy.to('kJ mol-1'), rxn.ts.enthalpy.to('kJ mol-1')
            results['G'][str(temp)] = {'forward': float(g_ts - g_reacs), 'reverse': float(g_ts - g_prods)}
            results['H'][str(temp)] = {'forward': float(h_ts - h_reacs), 'reverse': float(h_ts - h_prods)}

        results['smarts'] = self.smarts
        if self.output_file_name is not None:
            with open(self.output_file_name, 'w') as f:
                json.dump(results, f)
            f.close()
        self.rxn_results = results

        return results


class Decomposition(TsSearchBase):
    """
    A class for decomposition transition state search
    """

    def __init__(self,
                 output_file_name: str,
                 reactants: dict,
                 products: dict,
                 dir_name: str = None,
                 hmethod: Literal['g16', 'orca'] = 'g16',
                 lmethod: Literal['xtb', 'mopac'] = 'xtb'):

        self.relative_energies = None
        self.scaned_distance = None
        self.reverse_reacs_prods = False

        if len(reactants.values()) > 1:
            if len(products.values()) == 1:
                reactants, products = products, reactants
                self.reverse_reacs_prods = True
            else:
                raise TypeError('This method only support unimolecular reaction')

        self.mapped_rsmis, self.mapped_psmis = [], []
        self.rsmis, self.psmis = [], []
        new_reacs, new_prods = dict({}), dict({})
        for name, value in reactants.items():
            mol = str_to_mol(value, explicit_hydrogens=True)
            for atom in mol.GetAtoms():
                if atom.GetAtomMapNum() == 0:
                    raise AttributeError('Reactant atoms should be mapped')
            self.mapped_rsmis.append(value)
            rsmi = drop_map_num(value)
            self.rsmis.append(rsmi)
            new_reacs[name] = rsmi

        for name, value in products.items():
            mol = str_to_mol(value, explicit_hydrogens=True)
            for atom in mol.GetAtoms():
                if atom.GetAtomMapNum() == 0:
                    raise AttributeError('Reactant atoms should be mapped')
            self.mapped_psmis.append(value)
            psmi = drop_map_num(value)
            self.psmis.append(psmi)
            new_prods[name] = psmi

        super().__init__(output_file_name, new_reacs, new_prods, dir_name, lmethod=lmethod, hmethod=hmethod)

        self.changed_idx = get_decomp_idx('.'.join(self.mapped_rsmis), '.'.join(self.mapped_psmis))
        if not self.changed_idx:
            raise TypeError('Unknown decomposition reaction')

    def ts_search(self,
                  save_fig: bool = True,
                  save_pes: bool = True,
                  rs: Union[Tuple[float, float], Tuple[float, float, float]] = None):
        """
        Scan reaction path
        Args:
            save_fig: saving figure of scan results
            save_pes: saving potential energy surface
            rs: reaction surface. A tuple contains (start point, step size, steps) or (step size, steps)

        Returns: thermochemistry of reactants, products, and points along reaction surface

        """
        file_path = os.path.join(self.initial_path, 'reacs_prods')
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        os.chdir(file_path)
        autode_obj = self._set_ade()
        calc_lmethod = getattr(autode_obj.methods, self.lmethod.upper())()
        calc_hmethod = getattr(autode_obj.methods, self.hmethod.upper())()
        autode_obj.Config.n_cores = self.sw_cfg[self.lmethod]['n_cores']
        autode_obj.Config.max_core = self.sw_cfg[self.hmethod]['max_core']
        hmethod_cores = self.sw_cfg[self.hmethod]['n_cores']

        reactants, products = self.reacs_prods_thermo(autode_obj,
                                                      calc_lmethod=calc_lmethod,
                                                      calc_hmethod=calc_hmethod,
                                                      hmethod_cores=hmethod_cores)

        # decomposition reaction
        reactant = reactants[0]

        reacs_e0 = float(reactant.energy.to('kJ mol-1') + reactant.zpe.to('kJ mol-1'))
        e0_prods = []
        for mol in products:
            e0_prods.append(float(mol.energy.to('kJ mol-1') + mol.zpe.to('kJ mol-1')))
        prods_e0 = np.sum(e0_prods)
        reacs_enthalpy = self.reac_prod_results['reactants'][0].h
        reacs_free_energy = self.reac_prod_results['reactants'][0].g
        prods_enthalpy = np.sum([p_thermo.h for p_thermo in self.reac_prod_results['products']], axis=0)
        prods_free_energy = np.sum([p_thermo.g for p_thermo in self.reac_prod_results['products']], axis=0)

        """======Following calculations utilize high code and thus setting n_cores of config======"""
        autode_obj.Config.n_cores = self.sw_cfg[self.hmethod]['n_cores']

        # scan reactant bond
        print(f'Scanning bonds between atom {self.changed_idx[0]} and atom {self.changed_idx[1]}...')
        file_path = os.path.join(self.initial_path, 'scan')
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        os.chdir(file_path)
        # Initialise a relaxed potential energy surface
        # Scan changed bond at 2 * changed_atom_distance
        print('Changed index: ', self.changed_idx)
        if rs is None:
            init_distance = reactant.distance(*self.changed_idx)
            steps = int(init_distance * 2 / 0.1)
            final_distance = init_distance + steps * 0.1
        elif len(rs) == 3:
            init_distance, steps = rs[0], rs[2]
            final_distance = init_distance + steps * rs[1]
        else:
            init_distance = reactant.distance(*self.changed_idx)
            steps = rs[1]
            final_distance = init_distance + steps * rs[0]

        pes = autode_obj.pes.RelaxedPESnD(species=reactant,
                                          rs={tuple(self.changed_idx): (init_distance, final_distance, steps + 1)})
        pes.calculate(method=calc_hmethod)  # using high method for relax scan
        # May we can scan using lmethod and re-optimize using high method !!!!!

        if save_fig:
            pes.plot('PES_relaxed.png')
        if save_pes:
            pes.save('pes.npz')

        # clac thermo of pes
        print('Calculating thermochemistry ')
        scan_results = dict({})
        scan_change_mult_results = dict({})
        init_mult = deepcopy(reactant.mult)
        init_name = deepcopy(reactant.name)
        for idx, xyz in tqdm(enumerate(pes._coordinates), total=pes._coordinates.shape[0]):
            scan_results[idx] = dict({})
            scan_change_mult_results[idx] = dict({})
            doc_name = 'thermo/{}'.format(idx)
            curr_path = os.path.join(self.initial_path, doc_name)
            if not os.path.exists(curr_path):
                os.makedirs(curr_path)
            os.chdir(curr_path)
            reactant.coordinates = xyz
            reactant.mult = init_mult
            reactant.name = init_name
            reactant.single_point(method=calc_hmethod)
            reactant.calc_thermo(method=calc_hmethod)
            scan_results[idx]['sp'] = float(reactant.energy.to("kJ mol-1"))
            scan_results[idx]['e0'] = float(reactant.energy.to("kJ mol-1") + reactant.zpe.to("kJ mol-1"))
            for temp in self.temps:
                reactant.calc_thermo(temp=temp, method=calc_hmethod)
                scan_results[idx][str(temp)] = {'g_cont': float(reactant.g_cont.to('kJ mol-1')),
                                                'h_cont': float(reactant.h_cont.to('kJ mol-1'))}

            imag_freq = [float(ifreq) for ifreq in reactant.imaginary_frequencies] if \
                reactant.imaginary_frequencies is not None else None
            scan_results[idx]['ifreq'] = min(imag_freq) if imag_freq is not None else None
            if imag_freq is not None and min(imag_freq) < -100:
                try:
                    frozen_distance = reactant.distance(*self.changed_idx)
                    reactant.mult = init_mult + 2
                    reactant.name = f'{init_name}_change_mult'
                    reactant._clear_energies_gradient_hessian()
                    reactant.constraints.distance = {tuple(self.changed_idx): frozen_distance}
                    reactant.optimise(method=calc_hmethod)
                    reactant.single_point(method=calc_hmethod)
                    reactant.calc_thermo(method=calc_hmethod)
                    scan_change_mult_results[idx]['sp'] = float(reactant.energy.to("kJ mol-1"))
                    scan_change_mult_results[idx]['e0'] = float(
                        reactant.energy.to("kJ mol-1") + reactant.zpe.to("kJ mol-1"))
                    for temp in self.temps:
                        reactant.calc_thermo(temp=temp, method=calc_hmethod)
                        scan_change_mult_results[idx][str(temp)] = {
                            'g_cont': float(reactant.g_cont.to('kJ mol-1')),
                            'h_cont': float(reactant.h_cont.to('kJ mol-1'))}
                    scan_change_mult_results[idx]['ifreq'] = min(imag_freq)
                except:
                    continue
        self.scaned_distance = [d for d in pes._rs[0]]
        # Change current file path
        os.chdir(self.initial_path)

        # merge thermo calculated at different spin multiplicity
        free_energy_surface, enthalpy_surface, e0_surface, ifreq_surface = [], [], [], []
        for temp in self.temps:
            temp_free_energy, temp_enthalpy, temp_e0, ifreq = [], [], [], []
            for idx in range(len(self.scaned_distance)):
                g = scan_results[idx]['sp'] + scan_results[idx][str(temp)]['g_cont']
                mult_g = 0
                if scan_change_mult_results[idx] != {}:
                    mult_g = scan_change_mult_results[idx]['sp'] + scan_change_mult_results[idx][str(temp)][
                        'g_cont']

                use_mult_energy = False if g < mult_g else True  # choose stable one
                temp_free_energy.append(mult_g if use_mult_energy else g)
                temp_enthalpy.append(scan_change_mult_results[idx]['sp'] +
                                     scan_change_mult_results[idx][str(temp)]['h_cont'] if use_mult_energy else
                                     scan_results[idx]['sp'] + scan_results[idx][str(temp)]['h_cont'])
                temp_e0.append(scan_change_mult_results[idx]['e0'] if use_mult_energy else scan_results[idx]['e0'])
                ifreq.append(scan_change_mult_results[idx]['ifreq'] if use_mult_energy else scan_results[idx]['ifreq'])

            free_energy_surface.append(temp_free_energy)
            enthalpy_surface.append(temp_enthalpy)
            e0_surface.append(temp_e0)
            ifreq_surface.append(ifreq)

        self.relative_energies = np.array(free_energy_surface) - np.min(free_energy_surface)

        # Analyzing TS with respect to temperature
        # Note that imaginary frequency and single point energy of
        # transition state are different at every temperature point
        if self.reverse_reacs_prods:
            rsmi, psmi = self.smarts.split('>>')[-1], self.smarts.split('>>')[0],
            self.rxn_results = {'G': {}, 'H': {}, 'ifreq': [], 'E0_reacs': prods_e0, 'E0_prods': reacs_e0,
                                'E0_TS': [], 'smarts': '>>'.join([rsmi, psmi])}
        else:
            self.rxn_results = {'G': {}, 'H': {}, 'ifreq': [], 'E0_reacs': reacs_e0, 'E0_prods': prods_e0,
                                'E0_TS': [], 'smarts': self.smarts}
        for temp, reac_g, reac_h, prod_g, prod_h, surface_g, surface_h, surface_e0, surface_ifreq in \
                zip(self.temps, reacs_free_energy, reacs_enthalpy, prods_free_energy, prods_enthalpy,
                    free_energy_surface, enthalpy_surface, e0_surface, ifreq_surface):
            idx = surface_g.index(max(surface_g))
            if self.reverse_reacs_prods:
                self.rxn_results['G'][str(temp)] = {'forward': surface_g[idx] - prod_g,
                                                    'reverse': surface_g[idx] - reac_g, }
                self.rxn_results['H'][str(temp)] = {'forward': surface_h[idx] - prod_g,
                                                    'reverse': surface_h[idx] - reac_g, }
            else:
                self.rxn_results['G'][str(temp)] = {'forward': surface_g[idx] - reac_g,
                                                    'reverse': surface_g[idx] - prod_g}
                self.rxn_results['H'][str(temp)] = {'forward': surface_h[idx] - reac_h,
                                                    'reverse': surface_h[idx] - prod_h}
            self.rxn_results['ifreq'].append(surface_ifreq[idx])
            self.rxn_results['E0_TS'].append(surface_e0[idx])
        with open('rxn_thermo.json', 'w') as f:
            json.dump(self.rxn_results, f)
        f.close()

    def plot_surface(self, filename: str = 'G_surface.png'):
        points = self.scaned_distance
        plot_1d_thermo(distance=points, temps=self.temps, energies=self.relative_energies, filename=filename)


class Migration(TsSearchBase):
    """
    A class for transition state calculation
    """

    def __init__(self,
                 output_file_name: str,
                 smarts: str,
                 dir_name: str = None,
                 hmethod: Literal['g16', 'orca'] = 'g16',
                 lmethod: Literal['xtb', 'mopac'] = 'xtb'):
        """
        Args:
            output_file_name:
            smarts:
            dir_name:
            hmethod:
            lmethod:
        """
        if not is_mapped(smarts):
            raise AtomMappingException("Incorrect atom mapping between reactants and products")
        self.mapped_reacs_list, self.mapped_prods_list = \
            smarts.split('>')[0].split('.'), smarts.split('>')[-1].split('.'),
        reactants = {f"reactant{idx}": drop_map_num(smi) for idx, smi in enumerate(self.mapped_reacs_list)}
        products = {f"product{idx}": drop_map_num(smi) for idx, smi in enumerate(self.mapped_prods_list)}
        super().__init__(output_file_name, reactants, products, dir_name, lmethod=lmethod, hmethod=hmethod)
        self.ade_ridxs = None
        assert len(reactants) == 2, 'Reactant should contain two molecules'

    def _get_complex(self, ade_obj: ade):

        # Find changed bonds
        rsmis_list, psmis_list = self.mapped_reacs_list, self.mapped_prods_list
        bbs_ini, fbs_ini = get_changed_bonds(rsmis_list, psmis_list)

        # ade complex
        rcomplex, ade_rsmis = initialize_autode_complex(smiles_list=rsmis_list, label='reacs', ade_obj=ade_obj)
        pcomplex, ade_psmis = initialize_autode_complex(smiles_list=psmis_list, label='prods', ade_obj=ade_obj)
        self.ade_ridxs = [[atom.GetAtomMapNum() for atom in str_to_mol(smi).GetAtoms()] for smi in ade_rsmis]

        # Mapping initial map number and ade indexes
        num_ade_idx_rmap = {}
        [num_ade_idx_rmap.update(match_atom_nums(smi1, smi2)) for smi1, smi2 in zip(rsmis_list, ade_rsmis)]
        num_ade_idx_pmap = {}
        [num_ade_idx_pmap.update(match_atom_nums(smi1, smi2)) for smi1, smi2 in zip(psmis_list, ade_psmis)]

        # transforming user-defined bond rearrangement to autode bond rearrangement
        # Reactant scan is performed, thus bbs and fbs are transformed with respect to rcomplex
        bbs = [(num_ade_idx_rmap[bb[0]], num_ade_idx_rmap[bb[1]]) for bb in bbs_ini]
        fbs = [(num_ade_idx_rmap[fb[0]], num_ade_idx_rmap[fb[1]]) for fb in fbs_ini]

        # Get broken bond distance and formation bond distance
        bbds = [float(rcomplex.distance(*bb)) for bb in bbs]
        fbds = [float(pcomplex.distance(num_ade_idx_pmap[fb[0]], num_ade_idx_pmap[fb[1]])) for fb in fbs_ini]

        bond_rearr = bond_rearrangement(forming_bonds=fbs, breaking_bonds=bbs)
        rcomplex = translate_rotate_reactant(reactant=rcomplex, bond_rearrangement=bond_rearr,
                                             shift_factor=1.5 if rcomplex.charge == 0 else 2.5)

        return [rcomplex, bbs, bbds], [pcomplex, fbs, fbds]

    @staticmethod
    def _get_scan_info(bbd, fbd, rs: List[tuple] = None):
        """
        Get scan information
        Args:
            bbd:
            fbd:
            rs:

        Returns:

        """

        if rs is None:
            bbd_start = bbd
            bbd_steps = int(bbd_start * 1 / 0.05)
            bbd_end = bbd_start + bbd_steps * 0.05

            fbd_start = fbd
            fbd_steps = int(fbd_start * 1 / 0.05)
            fbd_end = fbd_start + bbd_steps * 0.05
        elif len(rs) == 1:
            if len(rs[0]) == 3:
                bbd_steps = fbd_steps = rs[0][2]
                bbd_start = fbd_start = rs[0][0]
                bbd_end = fbd_end = bbd_start + bbd_steps * rs[0][1]
            else:
                bbd_steps = fbd_steps = rs[0][1]
                bbd_start = bbd
                fbd_start = fbd

                bbd_end = bbd_start + bbd_steps * rs[0][0]
                fbd_end = fbd_start + fbd_steps * rs[0][0]
        elif len(rs) == 2:
            def get_rs(info, bd):
                if len(info) == 3:
                    start, steps = info[0], info[2]
                    end = start + steps * info[1]
                else:
                    start, steps = bd, info[1]
                    end = start + steps * info[0]
                return start, end, steps

            bbd_start, bbd_end, bbd_steps = get_rs(rs[0], bbd)
            fbd_start, fbd_end, fbd_steps = get_rs(rs[1], fbd)
        else:
            raise KeyError('Unknown rs information')

        return (bbd_start, bbd_end, bbd_steps + 1), (fbd_start, fbd_end, fbd_steps + 1)

    @staticmethod
    def _pes_ts_search(x: list, y: list, energies: np.array, interp_factor: int = 4, tolerance: float = 0.01):
        """
        Second partial derivative test for saddle point search
        (https://en.wikipedia.org/wiki/Second_partial_derivative_test)
        H = [[dy / dr1dr1, dy / dr1dr2],
             [dy / dr2dr1, dy / dr2dr2]]
        D = dy / dr1dr1 * dy / dr2dr2 -  dy / dr2dr1 * dy / dr1dr2

        saddle point: 1. dy / dr1 = dy / dr1 = 0
                      2. D < 0

        Args:
            x: x distance list
            y: y distance list
            energies: energy matrix
            interp_factor: interpolation factor
            tolerance:

        Returns: ts coordinate

        """
        dr1, dr2 = np.gradient(energies, x, y)
        dr1dr1, dr1dr2 = np.gradient(dr1, x, y)
        dr2dr1, dr2dr2 = np.gradient(dr2, x, y)
        d_func = Smooth().interp_2d_func(x, y, dr1dr1 * dr2dr2 - dr2dr1 * dr1dr2)

        # dy / dr1 = dy / dr1 = 0
        dydr1 = get_xy_value(x_value=x, y_value=y, surface=dr1, contour_height=[0], interp_factor=interp_factor)
        dydr2 = get_xy_value(x_value=x, y_value=y, surface=dr2, contour_height=[0], interp_factor=interp_factor)

        # Get possible cross point of dy / dr1 = dy / dr1 = 0
        ini_coor = np.array([x[0], y[0]])
        bond_distance = []
        nearest_coor = []
        distances = []
        for name, values in dydr1.items():
            value1 = np.concatenate(values)
            value2 = np.concatenate(dydr2[name])
            for v1 in value1:
                for v2 in value2:
                    if d_func(*v1) < 0 or d_func(*v2) < 0:
                        error = np.sqrt(np.sum((v1 - v2) ** 2))
                        if error <= tolerance * np.sum(v1) / 2 or error <= tolerance * np.sum(v2) / 2:
                            d = (v1 + v2) / 2
                            bond_distance.append(d)
                            distances.append(np.sqrt(np.sum((d - ini_coor) ** 2) / 2))
                            # Get nearest distance and coordinate
                            nd = float('inf')
                            nc = [0, 0]
                            for idx1, r1 in enumerate(x):
                                for idx2, r2 in enumerate(y):
                                    coor_distance = np.sqrt(np.sum((np.array([r1, r2]) - d) ** 2))
                                    if coor_distance < nd:
                                        nc = [idx1, idx2]
                                        nd = coor_distance
                            nearest_coor.append(nc)
        if distances:
            distances, bond_distance, nearest_coor = zip(*sorted(zip(distances, bond_distance, nearest_coor),
                                                                 key=lambda x: x[0]))

        return bond_distance, nearest_coor

    @staticmethod
    def _find_near_coor(bbd, fbd, x, y, num: int = 1):
        infos = []
        for idx1, r1 in enumerate(x):
            for idx2, r2 in enumerate(y):
                d2 = (r1 - bbd) ** 2 + (r2 - fbd) ** 2
                infos.append([d2, (idx1, idx2)])

        infos = sorted(infos, key=lambda x: x[0])

        return infos[:num]

    @staticmethod
    def _get_trans_geom(complex_obj: ade.Molecule, indexes: List[list], bb: tuple, fb: tuple, bbd: float, fbd: float):
        """
        getting new geometry by given origin coordinate and final bond distance
        Args:
            complex_obj: complex object
            indexes: atom index lists of different molecules
            bb: atom index of broken bond
            fb: atom index of formation bond
            bbd: final distance of broken bond
            fbd: final distance of formation bond

        Returns: new coordinate

        """
        coordinate = np.array(complex_obj.coordinates)
        # bond: (migrated atom, other atom)
        migrated_atom_index = bb[0] if bb[0] in fb else bb[1]
        bb = bb if bb[0] in fb else (bb[1], bb[0])
        fb = fb if fb[0] in bb else (fb[1], fb[0])

        bb_vector = coordinate[bb[1]] - coordinate[bb[0]]
        fb_vector = coordinate[fb[1]] - coordinate[bb[0]]
        print('Migrate atom index: ', migrated_atom_index)
        if migrated_atom_index in indexes[0]:
            indexes[0].remove(migrated_atom_index)
            bb_mol_idx, fb_mol_idx = indexes[0], indexes[1]
        else:
            indexes[1].remove(migrated_atom_index)
            bb_mol_idx, fb_mol_idx = indexes[1], indexes[0]
        bb_mol_coor, fb_mol_coor = coordinate[bb_mol_idx], coordinate[fb_mol_idx]

        # do translation
        bb_trans_distance = bbd - float(complex_obj.distance(*bb))
        new_bb_mol_coor = translation(bb_mol_coor, vector=bb_vector, distance=bb_trans_distance)
        fb_trans_distance = fbd - float(complex_obj.distance(*fb))
        new_fb_mol_coor = translation(fb_mol_coor, vector=fb_vector, distance=fb_trans_distance)

        new_coor = np.zeros(shape=coordinate.shape)
        new_coor[[bb_mol_idx]] = new_bb_mol_coor
        new_coor[[fb_mol_idx]] = new_fb_mol_coor
        new_coor[migrated_atom_index] = coordinate[migrated_atom_index]

        return new_coor

    def ts_search(self,
                  save_fig: bool = True,
                  save_pes: bool = True,
                  rs: List[Union[Tuple[float, float], Tuple[float, float, float]]] = None,
                  max_search_steps: int = 5,
                  search_step_size: float = 0.01,
                  max_calc_num: int = 1,
                  pre_opt_ts: Literal['g16', 'orca'] = None,
                  ts_from: Literal['pes_geom', 'pes_distance', 'empirical'] = 'pes_geom'):

        """
        TS searching
        Args:
            save_fig: plotting and saving pes
            save_pes: saving pes results
            rs: reaction surface. A list tuple contains (start point, step size, steps) or (step size, steps)
                for breaking and formation bonds, If only contains one tuple, same value will be set for broken
                and formation bonds
            max_search_steps: the maximum search steps of bonds
            search_step_size: the step size of every step (A)
            max_calc_num: the maximum number of transition state to be calculated of each search steps
            pre_opt_ts: reactions can be pre-optimized using this software
            ts_from: ts guess methods

        Returns: None
        """
        autode_obj = self._set_ade()
        calc_lmethod = getattr(autode_obj.methods, self.lmethod.upper())()
        calc_hmethod = getattr(autode_obj.methods, self.hmethod.upper())()
        if pre_opt_ts is not None:
            pre_sw = getattr(ade.methods, pre_opt_ts.upper())()
            pre_key = pre_sw.keywords.opt_ts
            pre_key.functional = functional_dict['wb97xd']
            pre_key.basis_set = def2svp
            pre_key.dispersion = None
        else:
            pre_key = None
            pre_sw = None

        autode_obj.Config.n_cores = self.sw_cfg[self.lmethod]['n_cores']
        autode_obj.Config.max_core = self.sw_cfg[self.hmethod]['max_core']
        hmethod_cores = self.sw_cfg[self.hmethod]['n_cores']

        # reactants and products calculation
        file_path = os.path.join(self.initial_path, 'reacs_prods')
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        os.chdir(file_path)
        reactants, products = self.reacs_prods_thermo(autode_obj,
                                                      calc_lmethod=calc_lmethod,
                                                      calc_hmethod=calc_hmethod,
                                                      hmethod_cores=hmethod_cores)

        # Guess TS geometry
        print('Breaking and formation bond scan...')
        pes_file_path = os.path.join(self.initial_path, 'mig_pes')
        if not os.path.exists(pes_file_path):
            os.makedirs(pes_file_path)
        os.chdir(pes_file_path)
        # Scan changed bond at 2 * changed_atom_distance
        rcomplex_info, pcomplex_info = self._get_complex(ade_obj=autode_obj)
        rcomplex, bbs, bbds = rcomplex_info[0], rcomplex_info[1], rcomplex_info[2]
        pcomplex, fbs, fbds = pcomplex_info[0], pcomplex_info[1], pcomplex_info[2]
        bb_scan_info, fb_scan_info = self._get_scan_info(bbd=bbds[0], fbd=fbds[0], rs=rs)

        pes = autode_obj.pes.RelaxedPESnD(species=rcomplex,
                                          rs={bbs[0]: bb_scan_info, fbs[0]: fb_scan_info})
        pes.calculate(method=calc_lmethod)
        if save_pes:
            pes.save(f"{self.dir_name}_pes.npz")

        r_x = list(pes._rs[0])  # broken bond
        r_y = list(pes._rs[1])
        energies = np.array(pes.relative_energies.to('kJ mol-1'))
        if save_fig:
            plot_2d(r_x=r_x, r_y=r_y,
                    energies=np.array(pes.relative_energies.to('kJ mol-1')),
                    interp_factor=2,
                    energy_units_name='kJ mol-1',
                    filename="surface_interpolated.png")

        # TS search
        print('TS guessing...')
        ts_guess_file_path = os.path.join(self.initial_path, 'ts_guess')
        if not os.path.exists(ts_guess_file_path):
            os.makedirs(ts_guess_file_path)
        os.chdir(ts_guess_file_path)
        autode_obj.Config.n_cores = self.sw_cfg[self.hmethod]['n_cores']
        tss_sp, tss = [], []
        new_complexes = []
        for i_iter in range(max_search_steps):
            if ts_from == 'pes_geom':
                bds, pes_idxes = self._pes_ts_search(x=r_x, y=r_y, energies=energies,
                                                     tolerance=search_step_size * (i_iter + 1))
                if not bds:
                    continue
                for idx, coor in enumerate(pes_idxes[:max_calc_num]):
                    # get xtb optimized xyz file
                    fn_list = os.listdir(pes_file_path)
                    xtb_file_name = fnmatch.filter(fn_list, f"*scan_{coor[0]}-{coor[1]}_optimised_xtb.xyz")
                    new_complex = deepcopy(rcomplex)
                    new_complex.name = f'reactants{i_iter}-{idx}'
                    print(f'load coordinates from scan_{coor[0]}-{coor[1]}_optimised_xtb.xyz')
                    new_complex.coordinates = load_coor_from_xyz_file(os.path.join(pes_file_path, xtb_file_name[0]))
                    new_complexes.append(new_complex)
            elif ts_from == 'pes_distance':
                bds, pes_idxes = self._pes_ts_search(x=r_x, y=r_y, energies=energies,
                                                     tolerance=search_step_size * (i_iter + 1))
                if not bds:
                    continue
                print(f"bond distance next {bds}")
                for idx, bd in enumerate(bds[:max_calc_num]):
                    print(f'Found TS with bonds {bd}')
                    print(f'breaking bonds {bbs}, formation bonds {fbs}')
                    print(f'bond distance {bd}')
                    new_coor = self._get_trans_geom(rcomplex, indexes=self.ade_ridxs,
                                                    bb=bbs[0], fb=fbs[0], bbd=bd[0], fbd=bd[1])
                    new_complex = deepcopy(rcomplex)
                    new_complex.coordinates = new_coor
                    new_complex.name = f'reactants_{i_iter}-{idx}'
                    new_complexes.append(new_complex)
            else:
                bb_start, fb_start = bb_scan_info[0] * (i_iter * 0.1 + 1.3), fb_scan_info[0] * (i_iter * 0.1 + 1.3)
                infos = self._find_near_coor(bbd=bb_start, fbd=fb_start, x=r_x, y=r_y, num=max_calc_num)
                for idx, info in enumerate(infos):
                    coor = info[1]
                    fn_list = os.listdir(pes_file_path)
                    xtb_file_name = fnmatch.filter(fn_list, f"*scan_{coor[0]}-{coor[1]}_optimised_xtb.xyz")
                    new_complex = deepcopy(rcomplex)
                    new_complex.name = f'reactants{i_iter}-{idx}'
                    print(f'load coordinates from scan_{coor[0]}-{coor[1]}_optimised_xtb.xyz')
                    new_complex.coordinates = load_coor_from_xyz_file(os.path.join(pes_file_path, xtb_file_name[0]))
                    new_complexes.append(new_complex)

            for idx, clx in enumerate(new_complexes):

                calc_ts = ade.Calculation(name=f'ts_{i_iter}-{idx}',
                                          molecule=clx,
                                          method=calc_hmethod,
                                          keywords=calc_hmethod.keywords.opt_ts,
                                          n_cores=hmethod_cores)
                try:
                    if pre_opt_ts is not None:
                        print(f'Try {idx + 1}th calculation of {i_iter}th search...')
                        pre_calc_ts = ade.Calculation(name=f'pre_opt_ts_{i_iter}-{idx}',
                                                      molecule=clx,
                                                      method=pre_sw,
                                                      keywords=pre_key,
                                                      n_cores=hmethod_cores)
                        clx.optimise(calc=pre_calc_ts, n_cores=hmethod_cores)
                        if clx.imaginary_frequencies is None:
                            continue
                        else:
                            if len(clx.imaginary_frequencies) != 1:
                                continue
                            else:
                                print(f'TS was found by pre_opt method {pre_opt_ts}')

                    clx.optimise(calc=calc_ts, n_cores=hmethod_cores)
                except:
                    continue
                if len(clx.imaginary_frequencies) == 1:
                    print(f'Transition state was found by {self.hmethod}')
                    tss.append(clx)
                    tss_sp.append(float(clx.energy.to('kJ mol-1')))
                    break
            if tss:
                break
        if not tss:
            raise Exception('TS was not found')

        # TS thermo
        print('TS calculation')
        ts_sp = min(tss_sp)
        ts = tss[tss_sp.index(ts_sp)]
        ts_thermo_dict = {'sp': ts_sp, 'ifreq': float(ts.imaginary_frequencies[0])}
        reacs_e0 = [thermo.sp + thermo.zpe for thermo in self.reac_prod_results['reactants']]
        prods_e0 = [thermo.sp + thermo.zpe for thermo in self.reac_prod_results['products']]
        self.rxn_results = {'G': {}, 'H': {}, 'ifreq': float(ts.imaginary_frequencies[0]),
                            'E0_reacs': float(np.sum(reacs_e0)),
                            'E0_prods': float(np.sum(prods_e0)),
                            'E0_TS': float(ts.energy.to('kJ mol-1') + ts.zpe.to('kJ mol-1')),
                            'smarts': self.smarts}
        for temperature in self.temps:
            ts.calc_thermo(method=calc_hmethod, temp=temperature, n_cores=hmethod_cores)
            ts_g_cont, ts_h_cont = float(ts.g_cont.to('kJ mol-1')), float(ts.h_cont.to('kJ mol-1'))
            ts_thermo_dict[str(temperature)] = {'g_cont': ts_g_cont, 'h_cont': ts_h_cont}
            # Get reactants and prods thermo
            reacs_g_thermo = [thermo.thermo_dict[temperature]['g'] for thermo in self.reac_prod_results['reactants']]
            reacs_h_thermo = [thermo.thermo_dict[temperature]['h'] for thermo in self.reac_prod_results['reactants']]

            prods_g_thermo = [thermo.thermo_dict[temperature]['g'] for thermo in self.reac_prod_results['products']]
            prods_h_thermo = [thermo.thermo_dict[temperature]['h'] for thermo in self.reac_prod_results['products']]

            self.rxn_results['G'][str(temperature)] = \
                {'forward': float(ts_g_cont) + float(ts_sp) - float(np.sum(reacs_g_thermo)),
                 'reverse': float(ts_g_cont) + float(ts_sp) - float(np.sum(prods_g_thermo))}

            self.rxn_results['H'][str(temperature)] = \
                {'forward': float(ts_h_cont) + float(ts_sp) - float(np.sum(reacs_h_thermo)),
                 'reverse': float(ts_h_cont) + float(ts_sp) - float(np.sum(prods_h_thermo))}

        with open(os.path.join(self.initial_path, 'ts_thermo.yaml'), 'w') as f:
            yaml.dump(ts_thermo_dict, f)
        f.close()

        with open(os.path.join(self.initial_path, 'rxn_thermo.yaml'), 'w') as f:
            yaml.dump(self.rxn_results, f)
        f.close()

        os.chdir(self.initial_path)


class TSCalc(TsSearchBase):
    """
    A class for transition state calculation
    Searching TST using given TS coordinates
    """

    def __init__(self,
                 output_file_name: str,
                 smarts: str,
                 dir_name: str = None,
                 hmethod: Literal['g16', 'orca'] = 'g16',
                 lmethod: Literal['xtb', 'mopac'] = 'xtb'):
        """
        Args:
            output_file_name:
            smarts:
            dir_name:
            hmethod:
            lmethod:
        """
        self.reacs_list, self.prods_list = \
            smarts.split('>')[0].split('.'), smarts.split('>')[-1].split('.'),
        reactants = {f"reactant{idx}": drop_map_num(smi) for idx, smi in enumerate(self.reacs_list)}
        products = {f"product{idx}": drop_map_num(smi) for idx, smi in enumerate(self.prods_list)}
        super().__init__(output_file_name, reactants, products, dir_name, lmethod=lmethod, hmethod=hmethod)
        self.ade_ridxs = None

    def ts_calc(self,
                ts_coor_path: str,
                mult: int = None,
                charge: int = None):

        """
        TS searching
        Args:
           ts_coor_path: path of ts guess geom

        Returns: None
        """
        autode_obj = self._set_ade()
        calc_lmethod = getattr(autode_obj.methods, self.lmethod.upper())()
        calc_hmethod = getattr(autode_obj.methods, self.hmethod.upper())()

        autode_obj.Config.n_cores = self.sw_cfg[self.lmethod]['n_cores']
        autode_obj.Config.max_core = self.sw_cfg[self.hmethod]['max_core']
        hmethod_cores = self.sw_cfg[self.hmethod]['n_cores']

        # reactants and products calculation
        file_path = os.path.join(self.initial_path, 'reacs_prods')
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        os.chdir(file_path)
        reactants, products = self.reacs_prods_thermo(autode_obj,
                                                      calc_lmethod=calc_lmethod,
                                                      calc_hmethod=calc_hmethod,
                                                      hmethod_cores=hmethod_cores)

        print('TS calculation')
        ts_guess_file_path = os.path.join(self.initial_path, 'ts_guess')
        if not os.path.exists(ts_guess_file_path):
            os.makedirs(ts_guess_file_path)
        os.chdir(ts_guess_file_path)
        print('load xyz file from file ', ts_coor_path)
        if mult is None:
            mult = CalculateSpinMultiplicity('.'.join(self.reacs_list))
        if charge is None:
            charge = 0
        ts_input_names = fnmatch.filter(os.listdir(), 'ts_*_g16.com')
        if ts_input_names:
            name_idxes = [int(name.split('.')[0].split('_')[1]) for name in ts_input_names]
            name_idx = 0
            while name_idx in name_idxes:
                name_idx += 1
        else:
            name_idx = 0
        ts_complex = ade.Species(atoms=xyz_file_to_atoms(ts_coor_path), name=f'ts_{name_idx}', charge=charge, mult=mult)
        # ts_complex.mult = CalculateSpinMultiplicity('.'.join(self.reacs_list))
        calc_ts = ade.Calculation(name=f'ts_{name_idx}',
                                  molecule=ts_complex,
                                  method=calc_hmethod,
                                  keywords=calc_hmethod.keywords.opt_ts,
                                  n_cores=hmethod_cores)

        ts_complex.optimise(calc=calc_ts, n_cores=hmethod_cores)

        # TS thermo
        ts_sp = ts_complex.energy.to('kJ mol-1')
        ts_thermo_dict = {'sp': ts_complex.energy.to('kJ mol-1'), 'ifreq': float(ts_complex.imaginary_frequencies[0])}
        reacs_e0 = [thermo.sp + thermo.zpe for thermo in self.reac_prod_results['reactants']]
        prods_e0 = [thermo.sp + thermo.zpe for thermo in self.reac_prod_results['products']]
        self.rxn_results = {'G': {}, 'H': {}, 'ifreq': float(ts_complex.imaginary_frequencies[0]),
                            'E0_reacs': float(np.sum(reacs_e0)),
                            'E0_prods': float(np.sum(prods_e0)),
                            'E0_TS': float(ts_complex.energy.to('kJ mol-1') + ts_complex.zpe.to('kJ mol-1')),
                            'smarts': self.smarts}
        for temperature in self.temps:
            ts_complex.calc_thermo(method=calc_hmethod, temp=temperature, n_cores=hmethod_cores)
            ts_g_cont, ts_h_cont = float(ts_complex.g_cont.to('kJ mol-1')), float(ts_complex.h_cont.to('kJ mol-1'))
            ts_thermo_dict[str(temperature)] = {'g_cont': ts_g_cont, 'h_cont': ts_h_cont}
            # Get reactants and prods thermo
            reacs_g_thermo = [thermo.thermo_dict[temperature]['g'] for thermo in self.reac_prod_results['reactants']]
            reacs_h_thermo = [thermo.thermo_dict[temperature]['h'] for thermo in self.reac_prod_results['reactants']]

            prods_g_thermo = [thermo.thermo_dict[temperature]['g'] for thermo in self.reac_prod_results['products']]
            prods_h_thermo = [thermo.thermo_dict[temperature]['h'] for thermo in self.reac_prod_results['products']]

            self.rxn_results['G'][str(temperature)] = \
                {'forward': float(ts_g_cont) + float(ts_sp) - float(np.sum(reacs_g_thermo)),
                 'reverse': float(ts_g_cont) + float(ts_sp) - float(np.sum(prods_g_thermo))}

            self.rxn_results['H'][str(temperature)] = \
                {'forward': float(ts_h_cont) + float(ts_sp) - float(np.sum(reacs_h_thermo)),
                 'reverse': float(ts_h_cont) + float(ts_sp) - float(np.sum(prods_h_thermo))}

        with open(os.path.join(self.initial_path, f'ts_thermo_{name_idx}.yaml'), 'w') as f:
            yaml.dump(ts_thermo_dict, f)
        f.close()

        with open(os.path.join(self.initial_path, f'rxn_thermo_{name_idx}.yaml'), 'w') as f:
            yaml.dump(self.rxn_results, f)
        f.close()

        os.chdir(self.initial_path)
