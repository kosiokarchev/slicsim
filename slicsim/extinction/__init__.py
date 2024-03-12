from abc import ABC
from dataclasses import dataclass
from enum import Enum

import torch

from phytorch.interpolate import LinearNDGridInterpolator

from ..utils import _t, DataRegistry
from ..utils.interpolated import DelayedInterpolated, Interpolated, DelayedLinear1dInterpolated


class Extinction(ABC):
    def mag(self, wave: _t, **kwargs) -> _t:
        return -2.5 * torch.log10(self.linear(wave, **kwargs))

    def linear(self, wave: _t, **kwargs) -> _t:
        return 10**(-0.4 * self.mag(wave, **kwargs))


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

    def _interpolate(self, wave: _t, *args, **kwargs) -> _t:
        return self._interp(LinearNDGridInterpolator.interp_input(wave, *args))


class DelayedLinearNDInterpolatedExtinction(
        LinearNDInterpolatedExtinction,
        DelayedInterpolated):
    pass


@dataclass
class Fitzpatrick99(DelayedLinearNDInterpolatedExtinction):
    Rv: _t = 3.1

    _delayed_data_func = DataRegistry.extinction_F99

    def _interpolate(self, wave: _t, **kwargs) -> _t:
        return super()._interpolate(wave, self.Rv)
