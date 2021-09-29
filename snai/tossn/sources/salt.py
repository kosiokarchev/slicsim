from __future__ import annotations

from typing import ClassVar, get_type_hints, Type

from .abc import ColouredSource, DelayedGridInterpPCASource
from ..extinction import DelayedLinear1dInterpolatedExtinction, Extinction
from ..utils import _t, DataRegistry


_data_funcT = get_type_hints(DelayedLinear1dInterpolatedExtinction)['_delayed_data_func']


def SALTExtinction(data_func: _data_funcT):
    class _Extinction(DelayedLinear1dInterpolatedExtinction):
        _mag_or_linear = DelayedLinear1dInterpolatedExtinction._MagOrLinear.mag
        _delayed_data_func = data_func

    return _Extinction


class SALTSource(ColouredSource, DelayedGridInterpPCASource):
    x_0: _t = 1.
    coeff_0 = property(lambda self: self.x_0)

    x_1: _t = 0.
    coeffs = property(lambda self: self.x_1)

    c: _t = 0.
    coeff_colour = property(lambda self: self.c)

    _Extinction: ClassVar[Type[Extinction]]

    def colourlaw(self, phase: _t, wave: _t) -> _t:
        return self._Extinction.linear(wave)


class SALT2Source(SALTSource):
    _delayed_data_func = DataRegistry.salt2_4
    _Extinction = SALTExtinction(DataRegistry.extinction_salt2_4)


class SALT3Source(SALTSource):
    _delayed_data_func = DataRegistry.salt3_k21
    _Extinction = SALTExtinction(DataRegistry.extinction_salt3_k21)

