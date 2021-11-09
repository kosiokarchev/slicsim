from abc import ABCMeta
from functools import cached_property, wraps
from typing import Callable, Protocol, Set, Union

import torch
from more_itertools import first
from phytorch.interpolate import Linear1dInterpolator
from phytorch.utils import _mid_many
from torch import Tensor

from ..utils import _t


class Descriptor(Protocol):
    def __get__(self, instance, owner): ...
    def __set__(self, instance, value): ...
    def __set_name__(self, owner, name): ...


class SpectralDensityMeta(ABCMeta):
    def __instancecheck__(self, instance):
        return issubclass(type(instance), self) or (
                issubclass(type(instance), TransformedSpectralDensity)
                and isinstance(instance.base, self))


class SpectralDensity(metaclass=SpectralDensityMeta):
    def derivative(self, wave: _t) -> _t: ...
    def integral(self, wave: _t) -> _t: ...

    def integrate(self, wave0: _t, wave1: _t) -> _t:
        return self.integral(wave1) - self.integral(wave0)

    def __call__(self, wave: _t) -> _t: ...

    _wave_like_attrs: Set[Union[str, Descriptor]] = set()
    _y_like_attrs: Set[Union[str, Descriptor]] = set()

    def __init_subclass__(cls, **kwargs):
        cls._wave_like_attrs, cls._y_like_attrs = ({
            arg if isinstance(arg, str)
            else first((key for key, val in cls.__dict__.items() if val == arg), arg)
            for arg in collection
        } for collection in (cls._wave_like_attrs, cls._y_like_attrs))


class TransformedSpectralDensity(SpectralDensity):
    base: SpectralDensity

    def __getattr__(self, item):
        attr = getattr(self.base, item)
        if item in self.base._wave_like_attrs:
            return self._transform_wave_like_attr(attr)
        if item in self.base._y_like_attrs:
            return self._transform_y_like_attr(attr)
        return attr

    _transform_wave_like_attr = _transform_y_like_attr = _transform_arg = staticmethod(lambda x: x)

    # noinspection PyUnusedLocal
    def transform_args(self, wrapped: Callable, *args, **kwargs):
        return ((*(self._transform_arg(arg) for arg in args),),
                {key: self._transform_arg(val) for key, val in kwargs})

    def transform(self, result, wrapped: Callable = None, *args, **kwargs):
        return result

    class TransformingOverride:
        def __set_name__(self, owner, name):
            wrapped = getattr(super(owner, owner), name)

            @wraps(wrapped)
            def f(slf: 'TransformedSpectralDensity', *args, **kwargs):
                args, kwargs = slf.transform_args(wrapped, *args, **kwargs)
                result = getattr(slf.base, name)(*args, **kwargs)
                return slf.transform(result, wrapped, *args, **kwargs)

            setattr(owner, name, f)

    # TODO: fix with stubs
    for name in ('derivative', 'integral', '__call__'):
        locals()[name] = TransformingOverride()


class ScaledSpectralDensity(TransformedSpectralDensity):
    scale: _t

    def transform(self, result, wrapped: Callable = None, *args, **kwargs):
        return self.scale * result

    _transform_y_like_attr = transform


class RedshiftedSpectralDensity(TransformedSpectralDensity):
    scale_factor: _t

    def _transform_y_like_attr(self, arg):
        return arg * self.scale_factor

    def _transform_wave_like_attr(self, attr):
        return attr / self.scale_factor

    _transform_arg = _transform_y_like_attr

    def transform(self, result, wrapped: Callable = None, *args, **kwargs):
        return (
            result * self.scale_factor if wrapped.__name__ == SpectralDensity.__call__.__name__ else
            result * self.scale_factor**2 if wrapped.__name__ == SpectralDensity.derivative.__name__
            else result
        )


class InterpolatedSpectralDensity(SpectralDensity):
    grid_wave: Tensor
    grid_y: Tensor

    def _build_integral(self) -> Callable[[_t], _t]:
        raise NotImplementedError

    @cached_property
    def _integral(self):
        return self._build_integral()

    def integral(self, wave: _t) -> _t:
        return self._integral(wave)


class LinearInterpolatedSpectralDensity(Linear1dInterpolator, InterpolatedSpectralDensity):
    @cached_property
    def grid_wave(self):
        return self.x

    @cached_property
    def grid_y(self):
        return self.y

    _wave_like_attrs = InterpolatedSpectralDensity._wave_like_attrs | {grid_wave}
    _y_like_attrs = InterpolatedSpectralDensity._y_like_attrs | {grid_y}

    def _build_integral(self) -> Linear1dInterpolator:
        y = (_mid_many(self.grid_y, (-1,)) * self.dx.squeeze(-1)).cumsum(-1)
        return Linear1dInterpolator(self.grid_wave, torch.cat((
            y.new_zeros(1).expand(*y.shape[:-1], 1), y)))

