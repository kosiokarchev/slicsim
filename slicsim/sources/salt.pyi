from typing import ClassVar, Type

from .abc import ColouredSource, DelayedGridInterpPCASource
from ..extinction import DelayedLinear1dInterpolatedExtinction, Extinction
from ..utils import _t


def SALTExtinction(data_func) -> Type[DelayedLinear1dInterpolatedExtinction]: ...


class SALTSource(ColouredSource, DelayedGridInterpPCASource):
    x_0: _t
    x_1: _t
    c: _t

    _Extinction: ClassVar[Type[Extinction]]

    def colourlaw(self, phase: _t, wave: _t) -> _t: ...

    def __call__(self, phase: _t, wave: _t, *, x_0: _t = 1., x_1: _t = 0., c: _t = 0., **kwargs): ...

class SALT2Source(SALTSource): ...
class SALT3Source(SALTSource): ...
