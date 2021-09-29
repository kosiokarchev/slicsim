from __future__ import annotations

from .abc import DelayedGridInterpSEDSource
from ..utils import _t, DataRegistry


class HsiaoSource(DelayedGridInterpSEDSource):
    A: _t = 1.
    coeff_0 = property(lambda self: self.A)

    _delayed_data_func = DataRegistry.hsiao
