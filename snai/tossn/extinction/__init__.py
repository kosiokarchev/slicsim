from abc import ABC, abstractmethod
from enum import Enum
from typing import Callable, ClassVar, Sequence, Type, Union

import torch
from phytorch.interpolate import Linear1dInterpolator, LinearNDGridInterpolator
from phytorch.interpolate.abc import AbstractBatchedInterpolator
from torch import Tensor

from ..utils import _t, cached_property, DataRegistry, Delayed


class Extinction(ABC):
    @classmethod
    def mag(cls, wave: _t, **kwargs) -> _t:
        return -2.5 * torch.log10(cls.linear(wave, **kwargs))

    @classmethod
    def linear(cls, wave: _t, **kwargs) -> _t:
        return 10**(-0.4 * cls.mag(wave, **kwargs))


class InterpolatedExtinction(Extinction, ABC):
    _interp: ClassVar[Callable[[Tensor], Tensor]]
    _interp_data: ClassVar[tuple[Union[Sequence[Tensor], Tensor], Tensor]]
    _interp_class: Type[AbstractBatchedInterpolator]

    @classmethod
    @cached_property
    def _interp(cls):
        return cls._interp_class(*cls._interp_data)

    @classmethod
    @abstractmethod
    def _interpolate(cls, wave: _t, *args) -> _t: ...

    class _MagOrLinear(Enum):
        mag = Extinction.mag
        linear = Extinction.linear

    _mag_or_linear: _MagOrLinear = _MagOrLinear.mag

    def __init_subclass__(cls, **kwargs):
        if hasattr(cls, '_mag_or_linear'):
            setattr(cls, cls._mag_or_linear.__name__, cls._interpolate)


class DelayedInterpolatedExtinction(InterpolatedExtinction, Delayed, ABC):
    _delayed_data_func: ClassVar[Callable[[], tuple[Union[Sequence[Tensor], Tensor], Tensor]]]
    _interp_data = Delayed.attribute(typ=tuple[Union[Sequence[Tensor], Tensor], Tensor])


class Linear1dInterpolatedExtinction(InterpolatedExtinction):
    _interp_class = Linear1dInterpolator

    @classmethod
    def _interpolate(cls, wave: _t, *args) -> _t:
        return cls._interp(wave)


class DelayedLinear1dInterpolatedExtinction(
        Linear1dInterpolatedExtinction,
        DelayedInterpolatedExtinction):
    pass


class FM07(DelayedLinear1dInterpolatedExtinction):
    _delayed_data_func = DataRegistry.extinction_FM07


class LinearNDInterpolatedExtinction(InterpolatedExtinction):
    _interp_class = LinearNDGridInterpolator

    @classmethod
    def _interpolate(cls, wave: _t, *args) -> _t:
        return cls._interp(LinearNDGridInterpolator.interp_input(wave, *args))


class DelayedLinearNDInterpolatedExtinction(
        LinearNDInterpolatedExtinction,
        DelayedInterpolatedExtinction):
    pass


class Fitzpatrick99(DelayedLinearNDInterpolatedExtinction):
    _delayed_data_func = DataRegistry.extinction_F99

    @classmethod
    def _interpolate(cls, wave: _t, Rv: _t = 3.1) -> _t:
        return super()._interpolate(wave, Rv)
