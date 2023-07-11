from abc import ABC, abstractmethod
from functools import cache
from typing import ClassVar, Mapping, NamedTuple, Type

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
    def zp_flux(cls, band: Bandpass): ...

    @classmethod
    @abstractmethod
    def zp_counts(cls, band: Bandpass): ...


class SpectralMagSys(MagSys):
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


class AB(SpectralMagSys):
    f0_freq: ClassVar = 10**(23 - 0.4*48.6) * jansky

    @classmethod
    def f0_wave(cls, wave):
        return cls.f0_freq * c / wave**2


class InterpolatedSpectralMagSys(DelayedLinear1dInterpolated, SpectralMagSys):
    _wave_unit: ClassVar = angstrom
    _flux_unit: ClassVar = erg / angstrom / second / centimeter**2

    @classmethod
    def f0_wave(cls, wave):
        return cls._interpolate(wave.to(cls._wave_unit).value) * cls._flux_unit


class Vega(InterpolatedSpectralMagSys):
    _delayed_data_func = DataRegistry.magsys_Vega


class BD17(InterpolatedSpectralMagSys):
    _delayed_data_func = DataRegistry.magsys_BD17


class CompositeMagSys(MagSys):
    class BandZP(NamedTuple):
        magsys: Type[MagSys]
        zp: float

    bands: ClassVar[Mapping[Bandpass, BandZP]]

    @classmethod
    @cache
    def zp_flux(cls, band: Bandpass):
        magsys, offset = cls.bands[band]
        return 10**(0.4*offset) * magsys.zp_flux(band)

    @classmethod
    @cache
    def zp_counts(cls, band: Bandpass):
        magsys, offset = cls.bands[band]
        return 10**(0.4*offset) * magsys.zp_counts(band)


class CSPK17MagSys(CompositeMagSys):
    @classmethod
    @property
    def bands(cls):
        from . import cspk17_r, cspk17_u, cspk17_g, cspk17_i, cspk17_B, cspk17_V0, cspk17_V1, cspk17_V, cspk17_Y, cspk17_J, cspk17_H, cspk17_Ydw, cspk17_Jdw, cspk17_Hdw

        return {
            cspk17_u: CompositeMagSys.BandZP(BD17, 10.518),
            cspk17_g: CompositeMagSys.BandZP(BD17, 9.644),
            cspk17_r: CompositeMagSys.BandZP(BD17, 9.352),
            cspk17_i: CompositeMagSys.BandZP(BD17, 9.250),
            cspk17_B: CompositeMagSys.BandZP(Vega, 0.030),
            cspk17_V0: CompositeMagSys.BandZP(Vega, 0.0096),
            cspk17_V1: CompositeMagSys.BandZP(Vega, 0.0145),
            cspk17_V: CompositeMagSys.BandZP(Vega, 0.0096),
            cspk17_Y: CompositeMagSys.BandZP(Vega, 0.),
            cspk17_J: CompositeMagSys.BandZP(Vega, 0.),
            cspk17_H: CompositeMagSys.BandZP(Vega, 0.),
            cspk17_Ydw: CompositeMagSys.BandZP(Vega, 0.),
            cspk17_Jdw: CompositeMagSys.BandZP(Vega, 0.),
            cspk17_Hdw: CompositeMagSys.BandZP(Vega, 0.)
        }
