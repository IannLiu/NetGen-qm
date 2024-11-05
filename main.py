# This is the main script module for netgen package
import os

import autode as ade
from qm.ts_search import Decomposition, TsSearch, Migration
from rdkit_util import check_rxn, is_mapped, drop_map_num
from rdkit import Chem
from rdkit.Chem import Descriptors
from typing import Literal, List, Tuple


class NetGenQM:
    """

    """

    def __init__(self,
                 rxn_smiles: str,
                 temps: List[float],
                 hmethod: Literal['orca', 'xtb', 'g16'] = 'orca',
                 lmethod: Literal['xtb', 'mopac'] = 'xtb',
                 lfm_method: Literal['igm', 'truhlar', 'grimme', 'minenkov'] = 'minenkov',
                 cal_free_energy: bool = True,
                 cal_enthalpy: bool = True
                 ):
        """

        Args:
            rxn_smiles: reaction smiles
            temps: reaction temperatures
            hmethod:
            lmethod:
            lfm_method: low frequency method
            cal_free_energy:
            cal_enthalpy:
        """
        if not is_mapped(rxn_smiles):
            rxn_type = 'ade_only'  # only can be calculated using autode
        else:
            rxn_type = 'others'
        rsmis, psmis = rxn_smiles.split('>>')[0], rxn_smiles.split('>>')[1]
        reactants = {f"reactant{idx}": rsmi for idx, rsmi in enumerate(rsmis.split('.'))}
        products = {f"product{idx}": psmi for idx, psmi in enumerate(psmis.split('.'))}
        rxn_wihout_map_num = f"{drop_map_num(rsmis)}>>{drop_map_num(psmis)}"
        output_file_name = rxn_wihout_map_num.replace('>>', '_to_').replace('.', '&')
        if rxn_type == 'ade_only':
            if Descriptors.NumRadicalElectrons(Chem.MolFromSmiles(rsmis)) != Descriptors.NumRadicalElectrons(
                    Chem.MolFromSmiles(psmis)):
                raise KeyError(
                    'The TS can not be founded using autode, and the input reaction SMILES should be atom-mapped')
            else:
                rxn_type = 'ade'
        else:
            rxn_type = check_rxn(rxn_smiles)
        self.output_file_name = output_file_name

        if rxn_type == 'ade':
            print('TS search using autode module')
            self.calc_rxn = TsSearch(reactants=reactants,
                                     products=products,
                                     output_file_name=output_file_name + '.json',
                                     dir_name=output_file_name,
                                     hmethod=hmethod,
                                     lmethod=lmethod)
        elif rxn_type == 'mig':
            print('TS search using migration module')
            self.calc_rxn = Migration(smarts=rxn_smiles,
                                      output_file_name=output_file_name + '.json',
                                      dir_name=output_file_name,
                                      hmethod=hmethod,
                                      lmethod=lmethod)
        elif rxn_type == 'decomp' or rxn_type == 'decomp_rev':
            print('TS search using decomposition module')
            self.calc_rxn = Decomposition(reactants=reactants,
                                          products=products,
                                          output_file_name=output_file_name + '.json',
                                          dir_name=output_file_name,
                                          hmethod=hmethod,
                                          lmethod=lmethod)
        else:
            raise KeyError('Unkonwn reaction type')

        self.calc_rxn.thermo_config(temps=temps,
                                    lfm_method=lfm_method,
                                    cal_free_energy=cal_free_energy,
                                    cal_enthalpy=cal_enthalpy)

    def sw_cms(self, cms: List[Tuple[str, int, float]]):
        """
        Set core and memory of every software
        Args:
            cms: [[software name, core numbers, Max memory of every core]]

        Returns: None

        """
        for cm in cms:
            self.calc_rxn.sw_config(name=cm[0], n_cores=cm[1], max_core=cm[2])

    def sw_level_of_theory(self,
                           slts: List[
                                      Tuple[Literal['sp', 'opt', 'opt_ts', 'hess'],
                                            Literal['orca', 'xtb', 'g16'],
                                            Literal['b3lyp', 'm062x', 'wb97xd', 'wb97xd3', 'wb97xd3bj', 'wb97mv',
                                                    'pwpb95', 'pw6b95', 'pw6b95d3', 'ccsdt'],
                                            Literal['def2tsvp', 'def2tzvp_f', 'def2tzvp', 'def2tzvpp', 'def2qzvp',
                                                    'def2qzvpp', 'madef2svp', 'madef2tzvp_f', 'madef2tzvp',
                                                    'madef2tzvpp', 'madef2qzvp', 'madef2qzvpp', 'ccpvdz', 'ccpvtz',
                                                    'ccpvqz', 'aug_ccpvdz', 'aug_ccpvtz','aug_ccpvqz', '631gx',
                                                    '631+gx', '6311gxx', '6311+gxx', 'def2svp'],
                                            Literal['d3bj', 'd3', 'no_dispersion']]
                                      ]):
        """
        Set level of theory
        Args:
            slts: the setting of level of theory [tasks type, software name, calculation method, basis set, dispersion]

        Returns: None
        """
        for slt in slts:
            self.calc_rxn.level_of_theory_config(calc_type=slt[0], sw_name=slt[1], functional=slt[2],
                                                 basis_set=slt[3], dispersion=slt[4])

    def get_kinetics(self):
        """
        Get kinetic parameters
        Returns: kinetic parameters

        """
        self.calc_rxn.ts_search()
        kins = self.calc_rxn.fit_by_arrhenius(save_fig=f"kin_{self.output_file_name + '.png'}",
                                              calc_reverse=True)

        os.chdir(os.path.dirname(self.calc_rxn.initial_path))
        return kins



