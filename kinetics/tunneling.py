from typing import Union, Tuple

import numpy as np
import scipy.constants as spc


class Wigner:
    """
    A tunneling model based on the Wigner formula.
    """

    def __init__(self, frequency: float):
        """

        Args:
            frequency: negative frequency of transition state default with unit cm^-1
        """
        self.frequency = frequency

    def __repr__(self):
        """
        Return a string representation of the tunneling model.
        """
        return 'Wigner(frequency={0!r})'.format(self.frequency)

    def calculate_tunneling_factor(self, temp: float):
        """
        Calculate and return the value of the Wigner tunneling correction for
        the reaction at the temperature `T` in K.
        """
        frequency = abs(self.frequency) * spc.speed_of_light * 100.0
        factor = spc.Planck * frequency / (spc.Boltzmann * temp)
        return 1.0 + factor * factor / 24.0


class Eckart:
    """
    A tunneling model based on the Eckart model.
    """

    def __init__(self,
                 frequency: float,
                 E0_reac: Union[float, Tuple[float, str]],
                 E0_TS: Union[float, Tuple[float, str]],
                 E0_prod: Union[float, Tuple[float, str]]=None):
        """
        Initialize Eckart model
         Args:
            frequency:     The imaginary frequency of the transition state
            E0_reac:       The ground-state energy of the reactants
            E0_TS:        The ground-state energy of the transition state
            E0_prod:       The ground-state energy of the products

        If `E0_prod` is not given, it is assumed to be the same as the reactants;
        this results in the so-called "symmetric" Eckart model. Providing
        `E0_prod`, and thereby using the "asymmetric" Eckart model, is the
        recommended approach.
        """
        self.frequency = frequency
        self.E0_reac = E0_reac
        self.E0_TS = E0_TS

        if E0_prod is None:
            self.E0_prod = self.E0_reac
        else:
            self.E0_prod = E0_prod

    def __repr__(self):
        """
        Return a string representation of the tunneling model.
        """
        return 'Eckart(frequency={0!r}, E0_reac={1!r}, E0_TS={2!r}, E0_prod={3!r})'.format(self.frequency, self.E0_reac,
                                                                                           self.E0_TS, self.E0_prod)

    def calculate_tunneling_factor(self, T):
        """
        Calculate and return the value of the Eckart tunneling correction for
        the reaction at the temperature `T` in K.
        """
        beta = 1. / (spc.gas_constant * T)  # [=] mol/J

        E0_reac = self.E0_reac
        E0_TS = self.E0_TS
        E0_prod = self.E0_prod

        # Calculate intermediate constants
        if E0_reac > E0_prod:
            E0 = E0_reac
            dV1 = E0_TS - E0_reac
            dV2 = E0_TS - E0_prod
        else:
            E0 = E0_prod
            dV1 = E0_TS - E0_prod
            dV2 = E0_TS - E0_reac

        if dV1 < 0 or dV2 < 0:
            raise ValueError('One or both of the barrier heights of {0:g} and {1:g} kJ/mol encountered in Eckart '
                             'method are invalid.'.format(dV1 / 1000., dV2 / 1000.))

        # Ensure that dV1 is smaller than dV2
        assert dV1 <= dV2

        # Evaluate microcanonical tunneling function kappa(E)
        dE = 100.
        Elist = np.arange(E0, E0 + 2. * (E0_TS - E0) + 40. * spc.gas_constant * T, dE)
        kappaE = self.calculate_tunneling_function(Elist)

        # Integrate to get kappa(T)
        kappa = np.exp(dV1 * beta) * np.sum(kappaE * np.exp(-beta * (Elist - E0))) * dE * beta

        # Return the calculated Eckart correction
        return kappa

    def calculate_tunneling_function(self, Elist: np.array):
        """
        Calculate and return the value of the Eckart tunneling function for
        the reaction at the energies `e_list` in J/mol.
        """

        frequency = abs(self.frequency) * spc.Planck * spc.speed_of_light * 100. * spc.Avogadro
        E0_reac = self.E0_reac
        E0_TS = self.E0_TS
        E0_prod = self.E0_prod

        _Elist = Elist

        # Calculate intermediate constants
        if E0_reac > E0_prod:
            E0 = E0_reac
            dV1 = E0_TS - E0_reac
            dV2 = E0_TS - E0_prod
        else:
            E0 = E0_prod
            dV1 = E0_TS - E0_prod
            dV2 = E0_TS - E0_reac

        # Ensure that dV1 is smaller than dV2
        assert dV1 <= dV2

        alpha1 = 2 * spc.pi * dV1 / frequency
        alpha2 = 2 * spc.pi * dV2 / frequency

        kappa = np.zeros_like(Elist)
        idx0 = 0
        for idx, E in enumerate(_Elist):
            if E >= E0:
                idx0 = idx
                break

        for r in range(idx0, _Elist.shape[0]):
            E = _Elist[r]

            xi = (E - E0) / dV1
            # 2 * pi * a
            twopia = 2. * np.sqrt(alpha1 * xi) / (1. / np.sqrt(alpha1) + 1. / np.sqrt(alpha2))
            # 2 * pi * b
            twopib = 2. * np.sqrt(abs((xi - 1.) * alpha1 + alpha2)) / (1. / np.sqrt(alpha1) + 1 / np.sqrt(alpha2))
            # 2 * pi * d
            twopid = 2. * np.sqrt(abs(alpha1 * alpha2 - 4 * spc.pi * spc.pi / 16.))

            # We use different approximate versions of the integrand to avoid
            # domain errors when evaluating cosh(x) for large x
            # If all of 2*pi*a, 2*pi*b, and 2*pi*d are sufficiently small,
            # compute as normal
            if twopia < 200. and twopib < 200. and twopid < 200.:
                kappa[r] = 1 - (np.cosh(twopia - twopib) + np.cosh(twopid)) / (np.cosh(twopia + twopib) + np.cosh(twopid))
            # If one of the following is true, then we can eliminate most of the
            # exponential terms after writing out the definition of cosh and
            # dividing all terms by exp(2*pi*d)
            elif twopia - twopib - twopid > 10 or twopib - twopia - twopid > 10 or twopia + twopib - twopid > 10:
                kappa[r] = 1 - np.exp(-2 * twopia) - np.exp(-2 * twopib) - np.exp(-twopia - twopib + twopid) - np.exp(
                    -twopia - twopib - twopid)
            # Otherwise expand each cosh(x) in terms of its exponentials and divide
            # all terms by exp(2*pi*d) before evaluating
            else:
                kappa[r] = 1 - (np.exp(twopia - twopib - twopid) + np.exp(-twopia + twopib - twopid) + 1 + np.exp(-2 * twopid)) / (
                            np.exp(twopia + twopib - twopid) + np.exp(-twopia - twopib - twopid) + 1 + np.exp(-2 * twopid))

        return kappa


if __name__ == "__main__":
    # Test Eckart
    print('Eckart tunneling factor test: ')
    frequency = -2017.96
    E0_reac = -295.563
    E0_TS = -12.7411
    E0_prod = (-10.2664) + (-253.48)
    tunneling = Eckart(frequency=frequency,
                       E0_reac=(E0_reac, 'kJ/mol'),
                       E0_TS=(E0_TS, 'kJ/mol'),
                       E0_prod=(E0_prod, 'kJ/mol'))
    Tlist = np.array([300, 500, 1000, 1500, 2000])
    kexplist = np.array([1623051., 7.69349, 1.46551, 1.18111, 1.09858])
    for T, kexp in zip(Tlist, kexplist):
        kact = tunneling.calculate_tunneling_factor(T)
        print('Calculated: {}   Test: {}'.format(kact, kexp))
    print("*" * 40)
    print('Test the Wigner tunneling factor')
    frequency = -2017.96
    tunneling = Wigner(
        frequency=frequency,
    )
    Tlist = np.array([300, 500, 1000, 1500, 2000])
    kexplist = np.array([4.90263, 2.40495, 1.35124, 1.15611, 1.08781])
    for T, kexp in zip(Tlist, kexplist):
        kact = tunneling.calculate_tunneling_factor(T)
        print('Calculated: {}   Test: {}'.format(kact, kexp))
    print("*" * 40)




