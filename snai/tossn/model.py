import dataclasses
from dataclasses import dataclass
from functools import cached_property
from typing import Iterable, Sequence, Union

from torch_scatter import segment_csr
from typing_extensions import TypeAlias
import numpy as np
import torch

from phytorch.constants import c, h
from phytorch.quantities import Quantity
from phytorch.units.si import angstrom, day

from .bandpasses.bandpass import Bandpass
from .sources.abc import Source
from .utils import _t


_times_T: TypeAlias = Sequence[Union[_t, '_times_T']]
_bands_T: TypeAlias = Sequence[Union[Bandpass, '_bands_T']]


@dataclass
class Field:
    times: _times_T
    bands: _bands_T

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
            # TODO: reduction produces quantity with None unit
            return segment_csr(t, self.indptr.view(*(t.ndim-1)*(1,), -1)).value * t.unit


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
                torch.atleast_1d(torch.as_tensor(self.times)).repeat_interleave(torch.tensor(sizes), dim=-1),
                waves, trans_dwaves
            ), (day, angstrom, angstrom))))


class LightcurveModel:
    def __init__(self, source: Source, field: Field, **kwargs):
        super().__init__(**kwargs)
        self.source = source
        self.field = field

    def _evaluate_points(self, **kwargs) -> Quantity:
        times, waves, trans_dwaves = self.field._evaluation_points
        # TODO: figure out a way to do heterogeneous-unit interp
        return (
            self.source(times.to(day).value, waves.to(angstrom).value, **kwargs)
            * self.source.flux_unit
            * trans_dwaves
        )

    def bandflux(self, **kwargs):
        return self.field._evaluation_points.reduce_add(self._evaluate_points(**kwargs))

    def bandcounts(self, **kwargs):
        return self.field._evaluation_points.reduce_add(
            self._evaluate_points(**kwargs) / self.field._evaluation_points.energies
        )
