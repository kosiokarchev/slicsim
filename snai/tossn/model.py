import dataclasses
from dataclasses import dataclass
from functools import cached_property
from typing import Iterable, Sequence

import torch

from phytorch.constants import c, h
from phytorch.quantities.torchquantity import TorchQuantity
from phytorch.ragged import RaggedQuantity
from phytorch.units.si import angstrom, day
from .bandpasses.Bandpass import Bandpass
from .sources.abc import Source
from .utils import _t


@dataclass
class Field:
    times: Sequence[_t]
    bands: Sequence[Bandpass]

    @dataclass
    class _evpT:
        times: RaggedQuantity
        waves: RaggedQuantity
        trans_dwaves: RaggedQuantity

        def __iter__(self) -> Iterable[RaggedQuantity]:
            return (getattr(self, f.name) for f in dataclasses.fields(self))

        _energies = None

        @property
        def energies(self):
            if self._energies is None:
                self._energies = h * c / self.waves
            return self._energies

    @cached_property
    def _evaluation_points(self):
        waves, trans_dwaves = zip(*((p.flatten() for p in (b.wave, b.trans_dwave)) for b in self.bands))
        sizes = tuple(map(len, waves))
        return self._evpT(*(
            RaggedQuantity(t, sizes=sizes, unit=u) for t, u in zip((
                torch.atleast_1d(torch.as_tensor(self.times)).repeat_interleave(torch.tensor(sizes), dim=-1),
                torch.cat(waves, -1),
                torch.cat(trans_dwaves, -1)
            ), (day, angstrom, angstrom))))


class LightcurveModel:
    def __init__(self, source: Source, field: Field):
        self.source = source
        self.field = field

    def _evaluate_points(self, **kwargs) -> RaggedQuantity:
        times, waves, trans_dwaves = self.field._evaluation_points
        # TODO: figure out a way to do heterogeneous-unit interp
        return TorchQuantity(self.source(times.to(day).value, waves.to(angstrom).value, **kwargs),
                             unit=self.source.flux_unit) * trans_dwaves

    def bandflux(self, **kwargs):
        return self._evaluate_points(**kwargs).reduce_add()

    def bandcounts(self, **kwargs):
        return (self._evaluate_points(**kwargs) / self.field._evaluation_points.energies).reduce_add()
