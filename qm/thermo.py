import numpy as np
from typing import Union, List


class ThermoMol:
    def __init__(self, name: str = None, single_point_energy: float = None, zpe: float = None):
        self.name = name
        self.sp = single_point_energy
        self.thermo_info = {}
        self.zpe = zpe

    def thermo_point(self,
                     temperature: Union[float, int],
                     g_cont: Union[float, int] = None,
                     h_cont: Union[float, int] = None):
        if temperature not in self.thermo_info.keys():
            self.thermo_info[temperature] = {'g_cont': g_cont, 'h_cont': h_cont}
        else:
            if g_cont is not None:
                self.thermo_info[temperature]['g_cont'] = g_cont
            if h_cont is not None:
                self.thermo_info[temperature]['h_cont'] = h_cont

    def thermo_points(self,
                      temperatures: List[Union[float, int]],
                      g_conts: List[Union[float, int]],
                      h_conts: List[Union[float, int]]
                      ):
        assert len(temperatures) == len(g_conts) == len(h_conts), 'Temperatures and thermo should have same length'
        for temp, g_cont, h_cont in zip(temperatures, g_conts, h_conts):
            self.thermo_point(temp, g_cont, h_cont)

    @property
    def g_cont(self):
        return [thermo['g_cont'] for temp, thermo in self.thermo_info.items()]

    @property
    def g(self):
        return [thermo['g_cont'] + self.sp for temp, thermo in self.thermo_info.items()]

    @property
    def h_cont(self):
        return [thermo['h_cont'] for temp, thermo in self.thermo_info.items()]

    @property
    def h(self):
        return [thermo['h_cont'] + self.sp for temp, thermo in self.thermo_info.items()]

    @property
    def temperatures(self):
        return list(self.thermo_info.keys())

    @property
    def thermo(self):
        return {'temperatures': self.temperatures, 'g': self.g, 'h_cont': self.h}

    @property
    def thermo_dict(self):
        return {temp: {'g': g, 'h': h} for temp, g, h in zip(self.temperatures, self.g, self.h)}


class ThermoTS(ThermoMol):
    def __init__(self, name: str = None, single_point_energy: float = None, ifreq: float = None):
        self.name = name
        self.ifreq = ifreq
        super().__init__(name=name, single_point_energy=single_point_energy)


if __name__ == "__main__":

    test = ThermoMol()
    test.thermo_point(1000, 19.5, 19.5)
    test.thermo_point(1100, 18.5, 18.5)
    print(test.temperatures)
    print(test.h_cont)
    print(test.g_cont)
    print(test.thermo_dict)
