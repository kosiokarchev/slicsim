from __future__ import annotations

from .abc import DelayedGridInterpSEDSource
from ..utils import _t, DataRegistry


class HsiaoSource(DelayedGridInterpSEDSource):
    # 1 / LightcurveModel(Distance(HsiaoSource()), Field([0.], [bessell_b], magsys=Vega)).bandcountscal()
    flux_unit = 1.1949464989803864e+40 * DelayedGridInterpSEDSource.flux_unit

    A: _t = 1.
    coeff_0 = property(lambda self: self.A)

    _delayed_data_func = DataRegistry.hsiao
