from abc import ABC, abstractmethod
from functools import cache
from typing import ClassVar

from phytorch.constants import c, h
from phytorch.units.astro import jansky
from phytorch.units.cgs import erg
from phytorch.units.si import centimeter, second, angstrom

from .bandpass import Bandpass
from ..utils import DataRegistry
from ..utils.interpolated import DelayedLinear1dInterpolated


class MagSys(ABC):
    @classmethod
    @abstractmethod
    def f0_wave(cls, wave): ...

    @classmethod
    @cache
    def zp_flux(cls, band: Bandpass):
        return (cls.f0_wave(band.uwave) * band.utrans_dwave).sum()

    @classmethod
    @cache
    def zp_counts(cls, band: Bandpass):
        return (cls.f0_wave(band.uwave) / (h*c/band.uwave) * band.utrans_dwave).sum()


class AB(MagSys):
    f0_freq: ClassVar = 10**(23 - 0.4*48.6) * jansky

    @classmethod
    def f0_wave(cls, wave):
        return cls.f0_freq * c / wave**2


class Vega(DelayedLinear1dInterpolated, MagSys):
    _delayed_data_func = DataRegistry.magsys_Vega

    _flux_unit: ClassVar = erg / angstrom / second / centimeter**2

    @classmethod
    def f0_wave(cls, wave):
        return cls._interpolate(wave.to(angstrom).value) * cls._flux_unit
