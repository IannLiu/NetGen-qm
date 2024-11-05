from typing import Dict, List
import numpy as np
from scipy.optimize import curve_fit
import scipy.constants

gas_constant = scipy.constants.gas_constant


def arrhenius(temperature, lnA, Ea):
    """
    Arrhenius equation
    Args:
        temperature: reaction temperature
        lnA: frequency factor
        Ea: energy barrier

    Returns: reaction rate

    """
    return lnA - Ea / gas_constant / temperature


def modified_arrhenius(temperature, lnA, n, Ea):
    """
    Modified Arrhenius equation
    Args:
        temperature: reaction temperature
        lnA: frequency factor
        n:
        Ea: energy barrier

    Returns:

    """
    return lnA + n * np.log(temperature) - Ea / gas_constant / temperature


def fit_kin_param(temperatures: List[float], rate_constants: List[float],  modified: bool = True):
    """
    Fit kinetic parameters which follow the Arrhenius form
    lnk = lnA + nlnT - E/R/T
    Args:
        temperatures: reaction temperatures
        rate_constants: reaction rate constants
        modified: Making explicit the temperature dependence of the pre-exponential factor

    Returns: kinetic parameters A, n, T

    """
    temperatures = np.array(temperatures)
    rate_cons = np.log(np.array(rate_constants))
    if modified:
        popt = curve_fit(modified_arrhenius, xdata=temperatures, ydata=rate_cons)
    else:
        popt = curve_fit(arrhenius, xdata=temperatures, ydata=rate_cons)

    popt[0][0] = np.exp(popt[0][0])

    return popt
