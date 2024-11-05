from autode.wrappers.keywords import Functional, DispersionCorrection, BasisSet, OptTSKeywords, OptKeywords
import autode as ade

from typing import List, Literal


functional_dict = {'b3lyp': Functional(name="b3lyp", orca="B3LYP", g16="B3LYP", freq_scale_factor=0.98),
                   'wb97xd': Functional(name="wb97xd", g16="wB97XD", freq_scale_factor=0.975),
                   'wb97xd3': Functional(name="wb97xd3", orca="wB97X-D3", freq_scale_factor=0.975),
                   'wb97xd3bj': Functional(name="wb97xd3bj", orca="wB97X-D3BJ", freq_scale_factor=0.975),
                   'm062x': Functional(name="m062x", orca="M06-2X", g16="M062X", freq_scale_factor=0.97),
                   'pw6b95d3': Functional(name="pw6b95d3", g16="PW6B95D3", freq_scale_factor=0.97),
                   'pw6b95': Functional(name="pw6b95", orca="PW6B95", g16="PW6B95", freq_scale_factor=0.97),
                   'wb97mv': Functional(name="wb97mv", orca="wB97M-V", freq_scale_factor=0.97),
                   'pwpb95': Functional(name="pwpb95", orca="PWPB95", freq_scale_factor=0.96),
                   'ccsdt': Functional(name="ccsd(t)", orca="CCSD(T)", g16="CCSD(T)", freq_scale_factor=0.99), }

base_set_dict = {
    '631gx': BasisSet(name="631gx", orca="6-31G*", g16="6-31G*"),
    '631+gx': BasisSet(name="631+gx", orca="6-31+G*", g16="6-31+G*"),
    '6311gxx': BasisSet(name="6311gxx", orca="6-311G**", g16="6-311G**"),
    '6311+gxx': BasisSet(name="6311+gxx", orca="6-311+G**", g16="6-311+G**"),
    'def2svp': BasisSet(name="def2svp", orca="def2-SVP", g16="Def2SVP"),
    'def2tzvp_f': BasisSet(name="def2tzvp(-f)", orca="def2-TZVP(-f)"),
    'def2tzvp': BasisSet(name="def2tzvp", orca="def2-TZVP", g16="Def2TZVP"),
    'def2tzvpp': BasisSet(name="def2tzvpp", orca="def2-TZVPP", g16="Def2TZVPP"),
    'def2qzvp': BasisSet(name="def2qzvp", orca="def2-QZVP", g16="Def2QZVP"),
    'def2qzvpp': BasisSet(name="def2qzvpp", orca="def2-QZVPP", g16="Def2QZVPP"),
    'madef2svp': BasisSet(name="madef2svp", orca="ma-def2-SVP"),
    'madef2tzvp_f': BasisSet(name="madef2tzvp(-f)", orca="ma-def2-TZVP(-f)"),
    'madef2tzvp': BasisSet(name="madef2tzvp", orca="ma-def2-TZVP"),
    'madef2tzvpp': BasisSet(name="madef2tzvpp", orca="ma-def2-TZVPP"),
    'madef2qzvp': BasisSet(name="madef2qzvp", orca="ma-def2-QZVP"),
    'madef2qzvpp': BasisSet(name="madef2qzvpp", orca="ma-def2-QZVPP"),
    'ccpvdz': BasisSet(name="cc-pvdz", orca="cc-pVDZ", g16="cc-pVDZ"),
    'ccpvtz': BasisSet(name="cc-pvtz", orca="cc-pVTZ", g16="cc-pVTZ"),
    'ccpvqz': BasisSet(name="cc-pvqz", orca="cc-pVQZ", g16="cc-pVQZ"),
    'aug_ccpvdz': BasisSet(name="cc-pvdz", orca="aug-cc-pVDZ", g16="aug-cc-pVDZ"),
    'aug_ccpvtz': BasisSet(name="cc-pvtz", orca="aug-cc-pVTZ", g16="aug-cc-pVTZ"),
    'aug_ccpvqz': BasisSet(name="cc-pvqz", orca="aug-cc-pVQZ", g16="aug-cc-pVQZ"),
}

dispersion_dict = {'no_dispersion': None,
                   'd3bj': DispersionCorrection(name="d3bj", orca="D3BJ", g16="GD3BJ"),
                   'd3': DispersionCorrection(name="d3", orca="D3", g16="GD3")}

func_str_dict = {
    'def2tzvp_f': 'BasisSet(name="def2tzvp(-f)", orca="def2-TZVP(-f)")',
    'b3lyp': 'Functional(name="b3lyp", orca="B3LYP", freq_scale_factor=0.98)',
    'ccsdt': 'Functional(name="ccsd(t)", orca="CCSD(T)", freq_scale_factor=0.99)',
    'wb97mv': 'Functional(name="wb97mv", orca="wB97M-V", freq_scale_factor=0.97)',
    'pwpb95': 'Functional(name="pwpb95", orca="PWPB95", freq_scale_factor=0.96)'}

base_set_str_dict = {
    'def2tzvp': 'BasisSet(name="def2-tzvp", orca="def2-TZVP")',
    'def2qzvpp': 'BasisSet(name="def2-qzvpp", orca="def2-QZVPP")',
    'def2tzvpp': 'BasisSet(name="def2-tzvpp", orca="def2-TZVPP")',
    'ccpvtz': 'BasisSet(name="cc-pvtz", orca="cc-pVTZ")',
    'ccpvqz': 'BasisSet(name="cc-pvqz", orca="cc-pVQZ")'}

dispersion_str_dict = {'no_dispersion': 'DispersionCorrection(name="no_dispersion", orca="")',
                       'd3bj': 'DispersionCorrection(name="d3bj", orca="D3BJ")',
                       'd3': 'DispersionCorrection(name="d3", orca="D3")'}

pbe0 = Functional(
    name="pbe0",
    doi_list=["10.1063/1.478522", "10.1103/PhysRevLett.77.3865"],
    orca="PBE0",
    g09="PBE1PBE",
    nwchem="pbe0",
    qchem="pbe0",
    freq_scale_factor=0.96,
)

pbe = Functional(
    name="pbe",
    doi_list=["10.1103/PhysRevLett.77.3865"],
    orca="PBE",
    g09="PBEPBE",
    nwchem="xpbe96 cpbe96",
    qchem="pbe",
    freq_scale_factor=0.99,
)

def2svp = BasisSet(
    name="def2-SVP",
    doi="10.1039/B508541A",
    orca="def2-SVP",
    g09="Def2SVP",
    nwchem="Def2-SVP",
    qchem="def2-SVP",
)

