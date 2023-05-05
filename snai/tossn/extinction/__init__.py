from abc import ABC
from enum import Enum

import torch

from phytorch.interpolate import LinearNDGridInterpolator

from ..utils import _t, DataRegistry
from ..utils.interpolated import DelayedInterpolated, Interpolated, DelayedLinear1dInterpolated


class Extinction(ABC):
    @classmethod
    def mag(cls, wave: _t, **kwargs) -> _t:
        return -2.5 * torch.log10(cls.linear(wave, **kwargs))

    @classmethod
    def linear(cls, wave: _t, **kwargs) -> _t:
        return 10**(-0.4 * cls.mag(wave, **kwargs))


class InterpolatedExtinction(Interpolated, Extinction, ABC):
    class _MagOrLinear(Enum):
        mag = Extinction.mag
        linear = Extinction.linear

    _mag_or_linear: _MagOrLinear = _MagOrLinear.mag

    def __init_subclass__(cls, **kwargs):
        if hasattr(cls, '_mag_or_linear'):
            setattr(cls, cls._mag_or_linear.__name__, cls._interpolate)


class FM07(DelayedLinear1dInterpolated, InterpolatedExtinction):
    _delayed_data_func = DataRegistry.extinction_FM07


class LinearNDInterpolatedExtinction(InterpolatedExtinction):
    _interp_class = LinearNDGridInterpolator

    @classmethod
    def _interpolate(cls, wave: _t, *args, **kwargs) -> _t:
        return cls._interp(LinearNDGridInterpolator.interp_input(wave, *args))


class DelayedLinearNDInterpolatedExtinction(
        LinearNDInterpolatedExtinction,
        DelayedInterpolated):
    pass


class Fitzpatrick99(DelayedLinearNDInterpolatedExtinction):
    _delayed_data_func = DataRegistry.extinction_F99

    @classmethod
    def _interpolate(cls, wave: _t, Rv: _t = 3.1, **kwargs) -> _t:
        return super()._interpolate(wave, Rv)
