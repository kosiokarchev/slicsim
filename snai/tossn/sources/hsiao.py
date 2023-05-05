from __future__ import annotations

from .abc import DelayedGridInterpSEDSource
from ..utils import _t, DataRegistry


class HsiaoSource(DelayedGridInterpSEDSource):
    # 1 / ((LightcurveModel(HsiaoSource(), Field([0.], [bessell_b])).bandcounts()
    #       / (4 * pi * (10 * pc)**2)
    #       ) / Vega.zp_counts(bessell_b)
    #      ).to(Unit()).value.item()
    flux_unit = 1.177175e+40 * DelayedGridInterpSEDSource.flux_unit

    A: _t = 1.
    coeff_0 = property(lambda self: self.A)

    _delayed_data_func = DataRegistry.hsiao
