import copyreg
from abc import ABC, abstractmethod, ABCMeta
from functools import cache, cached_property
from inspect import getattr_static
from typing import ClassVar, Mapping, NamedTuple, Iterable

from torch import Tensor

import phytorchx
from phytorch.constants import c, h
from phytorch.units.astro import jansky
from phytorch.units.cgs import erg
from phytorch.units.si import centimeter, second, angstrom
from .bandpass import Bandpass
from ..utils.interpolated import Linear1dInterpolated


class MagSys(ABC):
    @abstractmethod
    def zp_flux(self, band: Bandpass): ...

    @abstractmethod
    def zp_counts(self, band: Bandpass): ...


class SpectralMagSys(MagSys):
    @abstractmethod
    def f0_wave(self, wave): ...

    @cache
    def zp_flux(self, band: Bandpass):
        return (self.f0_wave(band.uwave) * band.utrans_dwave).sum()

    @cache
    def zp_counts(self, band: Bandpass):
        return (self.f0_wave(band.uwave) / (h*c/band.uwave) * band.utrans_dwave).sum()


class _AB(SpectralMagSys):
    f0_freq: ClassVar = 10**(23 - 0.4*48.6) * jansky

    def f0_wave(self, wave):
        return self.f0_freq * c / wave**2


AB = _AB()


class InterpolatedSpectralMagSys(Linear1dInterpolated, SpectralMagSys):
    _wave_unit = angstrom
    _flux_unit = erg / angstrom / second / centimeter**2

    _interp_data: tuple[Tensor, Tensor]
    _interpolate = Linear1dInterpolated._interpolate.__func__
    _interp = getattr_static(Linear1dInterpolated, '_interp').__func__

    def __init__(self, name, wave, flux):
        self.name = name
        self._interp_data = (wave, flux)

    def __repr__(self):
        return f'{type(self).__name__}[{self.name}]'

    def f0_wave(self, wave):
        return self._interpolate(wave.to(self._wave_unit).value) * self._flux_unit


def __getattr__(name):
    from ..utils import datadir

    fname = (datadir / 'magsys' / name).with_suffix('.pt')
    if fname.is_file():
        res = globals()[name] = InterpolatedSpectralMagSys(name, *phytorchx.load(fname))
        return res

    raise AttributeError(name)


Vega: InterpolatedSpectralMagSys = __getattr__('alpha_lyr_mod_001')
BD17: InterpolatedSpectralMagSys


class PicklableMagSysMeta(ABCMeta):
    _picklekeys: Iterable[str]

    def __getstate__(self):
        return {key: getattr(self, key) for key in self._picklekeys}

    def __reduce__(self):
        return type(self), (self.__name__, self.__bases__, type(self).__getstate__(self))


copyreg.pickle(PicklableMagSysMeta, PicklableMagSysMeta.__reduce__)


class CompositeMagSys(MagSys):
    class BandZP(NamedTuple):
        magsys: MagSys
        zp: float

    def __init__(self, name, bands: Mapping[Bandpass, BandZP]):
        self.name = name
        self.bands = bands

    def __repr__(self):
        return f'{type(self).__name__}[{self.name}]'

    @cache
    def zp_flux(self, band: Bandpass):
        magsys, offset = self.bands[band]
        return 10**(0.4*offset) * magsys.zp_flux(band)

    @cache
    def zp_counts(self, band: Bandpass):
        magsys, offset = self.bands[band]
        return 10**(0.4*offset) * magsys.zp_counts(band)


def CSPMagSys_K17():
    from . import cspk17_r, cspk17_u, cspk17_g, cspk17_i, cspk17_B, cspk17_V0, cspk17_V1, cspk17_V, cspk17_Y, cspk17_J, cspk17_Jrc2, cspk17_H, cspk17_Ydw, cspk17_Jdw, cspk17_Hdw

    BD17 = globals()['BD17'] if 'BD17' in globals() else __getattr__('BD17')

    return CompositeMagSys('CSPMagSys_K17', {
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
        cspk17_Jrc2: CompositeMagSys.BandZP(Vega, 0.),
        cspk17_H: CompositeMagSys.BandZP(Vega, 0.),
        cspk17_Ydw: CompositeMagSys.BandZP(Vega, 0.),
        cspk17_Jdw: CompositeMagSys.BandZP(Vega, 0.),
        cspk17_Hdw: CompositeMagSys.BandZP(Vega, 0.)
    })


def CSPMagSys_SNANA():
    from . import cspk17_r, cspk17_u, cspk17_g, cspk17_i, cspk17_B, cspk17_V0, cspk17_V1, cspk17_V, cspk17_Y, cspk17_J, cspk17_Jrc2, cspk17_H, cspk17_Ydw, \
        cspk17_Hdw

    BD17 = globals()['BD17'] if 'BD17' in globals() else __getattr__('BD17')

    return CompositeMagSys('CSPMagSys_SNANA', {
        # ZPs from SNANA:
        cspk17_u: CompositeMagSys.BandZP(BD17, 10.518),
        cspk17_g: CompositeMagSys.BandZP(BD17, 9.644),
        cspk17_r: CompositeMagSys.BandZP(BD17, 9.352),
        cspk17_i: CompositeMagSys.BandZP(BD17, 9.250),
        cspk17_B: CompositeMagSys.BandZP(BD17, 9.896),
        cspk17_V0: CompositeMagSys.BandZP(BD17, 9.492),
        cspk17_V1: CompositeMagSys.BandZP(BD17, 9.488),
        cspk17_V: CompositeMagSys.BandZP(BD17, 9.494),
        cspk17_Y: CompositeMagSys.BandZP(BD17, 8.632),
        cspk17_J: CompositeMagSys.BandZP(BD17, 8.419),
        cspk17_Jrc2: CompositeMagSys.BandZP(BD17, 8.426),
        cspk17_H: CompositeMagSys.BandZP(BD17, 8.125),
        cspk17_Ydw: CompositeMagSys.BandZP(BD17, 8.620),
        cspk17_Hdw: CompositeMagSys.BandZP(BD17, 8.126),
    })


def CSPMagSys_BayeSN():
    from . import cspk17_r, cspk17_u, cspk17_g, cspk17_i, cspk17_B, cspk17_V0, cspk17_V1, cspk17_V, cspk17_Y, cspk17_J, cspk17_Jrc2, cspk17_H, cspk17_Ydw, cspk17_Jdw, cspk17_Hdw

    _Vega = __getattr__('alpha_lyr_mod_004')

    return CompositeMagSys('CSPMagSys_BayeSN', {
        # ZPs from bayesn-public:
        cspk17_u: CompositeMagSys.BandZP(AB, 0),
        cspk17_g: CompositeMagSys.BandZP(AB, 0),
        cspk17_r: CompositeMagSys.BandZP(AB, 0),
        cspk17_i: CompositeMagSys.BandZP(AB, 0),
        cspk17_B: CompositeMagSys.BandZP(_Vega, 0),
        cspk17_V0: CompositeMagSys.BandZP(_Vega, 0),
        cspk17_V1: CompositeMagSys.BandZP(_Vega, 0),
        cspk17_V: CompositeMagSys.BandZP(_Vega, 0),
        cspk17_Y: CompositeMagSys.BandZP(_Vega, 0),
        cspk17_J: CompositeMagSys.BandZP(_Vega, 0),
        cspk17_Jrc2: CompositeMagSys.BandZP(_Vega, 0),
        cspk17_H: CompositeMagSys.BandZP(_Vega, 0),
        cspk17_Ydw: CompositeMagSys.BandZP(_Vega, 0),
        cspk17_Jdw: CompositeMagSys.BandZP(_Vega, 0.),
        cspk17_Hdw: CompositeMagSys.BandZP(_Vega, 0),
    })
