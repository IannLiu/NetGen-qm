import scipy.constants as spc
from typing import Union, Tuple
import numpy as np
from kinetics.tunneling import Eckart, Wigner

kb = spc.Boltzmann
planck = spc.Planck
p0 = 100000  # 1bar = 100 kPa = 100 000 Pa


def arrhenius_kinetics(freq_factor: float,
                       temp_factor,
                       activation_energy: float,
                       temperature: float):
    """
    Calculating the  kinetics by arrhenius equation
    Args:
        freq_factor: frequency factor, s-1 or unimolecule, mol-1 s-1 for bimolecule reaction
        temp_factor: float or float with units
        activation_energy: activation energy, kJ/mol
        temperature: reaction temperature, K

    Returns: rate constant in SI
    """
    rate_constant = freq_factor * np.power(temperature, temp_factor) * \
                     np.exp(-activation_energy / spc.gas_constant / temperature)

    return rate_constant


def tst_kinetics(smarts: Union[str, int],
                 free_energy: float,
                 temperature: float,
                 ifreq: float = None,
                 E0_reac: float = None,
                 E0_TS: float = None,
                 E0_prod: float = None
                 ):
    """
    Calculating the  kinetics by transition theory
    Args:
        free_energy: reaction free energy, kJ/mol
        smarts: reaction smarts or the number of reactants
        temperature: reaction temperature, K
        ifreq: imaginary frequency of transition state
        E0_reac: The ground-state energy of the reactants (EE+ZPE)
        E0_TS: The ground-state energy of the transition state (EE+ZPE)
        E0_prod: The ground-state energy of the products (EE+ZPE)

    Returns: rate constant in SI

    """

    pressure_factor = len(smarts.split('>>')[0].split('.')) if type(smarts).__name__ == 'str' else smarts
    pref = kb * temperature / planck * (spc.gas_constant * temperature / p0) ** (pressure_factor - 1)
    rate = np.exp(-free_energy * 1000 / spc.gas_constant / temperature) * pref
    if ifreq is not None:
        if E0_reac is not None and E0_TS is not None:
            tunneling = Eckart(frequency=ifreq, E0_reac=E0_reac, E0_TS=E0_TS, E0_prod=E0_prod)
            tunnel_factor = tunneling.calculate_tunneling_factor(temperature)
        else:
            tunneling = Wigner(frequency=ifreq)
            tunnel_factor = tunneling.calculate_tunneling_factor(temperature)
    else:
        tunnel_factor = 1

    return rate * tunnel_factor

