from snai.tossn.extinction import FM07

from .abc import ColouredSource, DelayedGridInterpPCASource
from ..utils import _t, DataRegistry


class SNEMOSource(ColouredSource, DelayedGridInterpPCASource):
    A_s: _t = 0.
    coeff_colour = property(lambda self: self.A_s)

    def colourlaw(self, phase: _t, wave: _t) -> _t:
        return FM07.linear(wave)


class SNEMO2Source(SNEMOSource):
    _delayed_data_func = DataRegistry.snemo_2


class SNEMO7Source(SNEMOSource):
    _delayed_data_func = DataRegistry.snemo_7


class SNEMO15Source(SNEMOSource):
    _delayed_data_func = DataRegistry.snemo_15
