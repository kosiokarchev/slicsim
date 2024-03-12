from dataclasses import dataclass
from functools import cached_property
from typing import Iterable, Mapping, Sequence, Union, cast

import numpy as np
import torch
from more_itertools import bucket
from torch import Tensor
from typing_extensions import TypeAlias

from phytorch.constants import c, h
from phytorch.quantities import Quantity
from phytorch.units import Unit
from phytorch.units.si import angstrom, day

from .bandpasses.bandpass import Bandpass
from .bandpasses.magsys import MagSys
from .sources.abc import Source
from .utils import _t

_times_T: TypeAlias = Sequence[Union[_t, '_times_T']]
_bands_T: TypeAlias = Sequence[Union[Bandpass, '_bands_T']]


@dataclass
class Field:
    times: _times_T
    bands: _bands_T
    magsys: MagSys

    @dataclass
    class _evpT:
        sizes: Sequence[int]
        times: Quantity
        waves: Quantity
        trans_dwaves: Quantity

        def __iter__(self) -> Iterable[Quantity]:
            return iter((self.times, self.waves, self.trans_dwaves))

        @cached_property
        def energies(self):
            return h * c / self.waves

        @cached_property
        def indptr(self):
            return torch.tensor([0] + [i for i in [0] for s in self.sizes for i in [i + s]],
                                device=self.times.device)

        def reduce_add(self, t):
            from torch_scatter import segment_csr

            # TODO: torch_scatter with units
            return segment_csr(t.value, self.indptr.view(*(t.ndim-1)*(1,), -1)) * t.unit


    @cached_property
    def _evaluation_points(self):
        waves, trans_dwaves = np.vectorize(
            lambda b: (b.wave, b.trans_dwave), otypes=(object, object))(self.bands)
        sizes = tuple(map(len, waves[(waves.ndim - 1) * (0,)]))
        waves, trans_dwaves = (
            torch.stack(tuple(a.flat), 0).reshape(a.shape + (-1,))
            for a in map(
                np.vectorize(lambda _: torch.cat(tuple(_), -1), otypes=(object,), signature='(n)->()'),
                (waves, trans_dwaves)
            )
        )
        return self._evpT(sizes, *(
            Quantity(t, unit=u) if not isinstance(t, Quantity) else t
            for t, u in zip((
                torch.atleast_1d(torch.as_tensor(self.times, dtype=torch.get_default_dtype())).repeat_interleave(torch.tensor(sizes), dim=-1),
                waves, trans_dwaves
            ), (day, angstrom, angstrom))))

    @cached_property
    def band_indices(self) -> Mapping[Bandpass, Sequence[int]]:
        return dict(zip(b := bucket(range(len(self.bands)), lambda i: self.bands[i]), (list(b[key]) for key in b)))

    @cached_property
    def band_zpfluxes(self) -> Quantity:
        return cast(Quantity, torch.stack([self.magsys.zp_flux(b) for b in self.bands]))

    @cached_property
    def band_zpcounts(self) -> Quantity:
        return cast(Quantity, torch.stack([self.magsys.zp_counts(b) for b in self.bands]))

    def cache(self, clear=False):
        if clear:
            del self._evaluation_points
            del self.band_indices
            del self.band_zpfluxes
            del self.band_zpcounts
        self._evaluation_points
        self.band_indices
        self.band_zpfluxes
        self.band_zpcounts
        return self



@dataclass
class LightcurveModel:
    source: Source
    field: Field

    def _evaluate_points(self, **kwargs) -> Quantity:
        times, waves, trans_dwaves = self.field._evaluation_points
        # TODO: figure out a way to do heterogeneous-unit interp
        return (
            self.source(times.to(day).value, waves.to(angstrom).value, **kwargs)
            * self.source.flux_unit
            * trans_dwaves
        )

    def bandflux(self, **kwargs) -> Quantity:
        return self.field._evaluation_points.reduce_add(self._evaluate_points(**kwargs))

    def bandfluxcal(self, **kwargs) -> Tensor:
        return (self.bandcounts(**kwargs) / self.field.band_zpfluxes).to(Unit()).value

    def bandcounts(self, **kwargs) -> Quantity:
        return self.field._evaluation_points.reduce_add(
            self._evaluate_points(**kwargs) / self.field._evaluation_points.energies
        )

    def bandcountscal(self, **kwargs) -> Tensor:
        return (self.bandcounts(**kwargs) / self.field.band_zpcounts).to(Unit()).value
