from abc import ABC, abstractmethod
from functools import cache

from phytorch.constants import c as speed_of_light
from phytorch.units.si import angstrom
from phytorch.units.astro import jansky

from .bandpass import Bandpass


class MagSys(ABC):
    @classmethod
    @abstractmethod
    def zeropoint(cls, band: Bandpass): ...


class AB(MagSys):
    f0_freq = 10**(23 - 0.4*48.6) * jansky

    @classmethod
    @cache
    def zeropoint(cls, band: Bandpass):
        return (cls.f0_freq * speed_of_light / band.wave**2 * band.trans_dwave / angstrom).sum()
