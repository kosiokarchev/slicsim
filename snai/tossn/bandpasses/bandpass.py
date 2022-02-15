from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from inspect import getattr_static

import torch
from phytorch.utils import _mid_many
from torch import Tensor

from ..extinction import Extinction, InterpolatedExtinction, Linear1dInterpolatedExtinction
from ..utils import _t, cached_property


@dataclass(unsafe_hash=True)
class Bandpass(Extinction):
    name: str

    _wave: Tensor = dataclasses.field(init=False, repr=False)
    _trans: Tensor = dataclasses.field(init=False, repr=False)

    @property
    def minwave(self) -> _t:
        return self._wave[..., 0]

    @property
    def maxwave(self) -> _t:
        return self._wave[..., -1]

    @cached_property
    def wave(self) -> Tensor:
        return _mid_many(self._wave, (-1,))

    @cached_property
    def trans(self) -> Tensor:
        return self.linear(self.wave)

    @cached_property
    def dwave(self) -> Tensor:
        return torch.diff(self._wave, dim=-1)

    @cached_property
    def trans_dwave(self) -> Tensor:
        return self.trans * self.dwave


# @dataclass
# class DiscretisedBandpass(Bandpass):
#     minwave: _t
#     maxwave: _t
#     dwave: _t
#
#     @property
#     def wave(self):
#         Dwave = self.maxwave - self.minwave
#         return self.minwave + torch.linspace(0., 1., (Dwave/self.dwave).max().item()) * Dwave


class InterpolatedBandpass(Bandpass, InterpolatedExtinction):
    _mag_or_linear = InterpolatedExtinction._MagOrLinear.linear


@dataclass(unsafe_hash=True)
class LinearInterpolatedBandpass(InterpolatedBandpass, Linear1dInterpolatedExtinction):
    _interp_data: tuple[Tensor, Tensor] = dataclasses.field(repr=False)
    # de-classmethod-ify...
    _interpolate = Linear1dInterpolatedExtinction._interpolate.__func__
    _interp = getattr_static(Linear1dInterpolatedExtinction, '_interp').__func__

    @property
    def _wave(self):
        return self._interp.x

    @property
    def _trans(self):
        return self._interp.y
