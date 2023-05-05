from abc import abstractmethod, ABC
from typing import ClassVar, Callable, Union, Sequence, Type

from torch import Tensor

from phytorch.interpolate import Linear1dInterpolator
from phytorch.interpolate.abc import AbstractBatchedInterpolator

from . import _t, cached_property, Delayed


class Interpolated(ABC):
    _interp: ClassVar[Callable[[Tensor], Tensor]]
    _interp_data: ClassVar[tuple[Union[Sequence[Tensor], Tensor], Tensor]]
    _interp_class: Type[AbstractBatchedInterpolator]

    @classmethod
    @cached_property
    def _interp(cls):
        return cls._interp_class(*cls._interp_data)

    @classmethod
    @abstractmethod
    def _interpolate(cls, *args, **kwargs) -> _t: ...


class DelayedInterpolated(Interpolated, Delayed, ABC):
    _delayed_data_func: ClassVar[Callable[[], tuple[Union[Sequence[Tensor], Tensor], Tensor]]]
    _interp_data = Delayed.attribute(typ=tuple[Union[Sequence[Tensor], Tensor], Tensor])


class Linear1dInterpolated(Interpolated):
    _interp_class = Linear1dInterpolator

    @classmethod
    def _interpolate(cls, arg: _t, *args, **kwargs) -> _t:
        return cls._interp(arg)


class DelayedLinear1dInterpolated(DelayedInterpolated, Linear1dInterpolated):
    pass
