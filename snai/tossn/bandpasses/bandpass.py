from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from inspect import getattr_static

import torch
from torch import Tensor

from phytorch.units.si import angstrom
from phytorchx import mid_many

from ..extinction import Extinction, InterpolatedExtinction
from ..utils import _t, cached_property
from ..utils.interpolated import Linear1dInterpolated


@dataclass(unsafe_hash=True)
class Bandpass(Extinction):
    name: str

    _wave: Tensor = dataclasses.field(init=False, repr=False, compare=False, hash=False)
    _trans: Tensor = dataclasses.field(init=False, repr=False, compare=False, hash=False)

    @property
    def minwave(self) -> _t:
        return self._wave[..., 0]

    @property
    def maxwave(self) -> _t:
        return self._wave[..., -1]

    @cached_property
    def wave(self) -> Tensor:
        return mid_many(self._wave, (-1,))

    @cached_property
    def trans(self) -> Tensor:
        return self.linear(self.wave)

    @cached_property
    def dwave(self) -> Tensor:
        return torch.diff(self._wave, dim=-1)

    @cached_property
    def trans_dwave(self) -> Tensor:
        return self.trans * self.dwave

    @cached_property
    def uwave(self) -> Tensor:
        return self.wave * angstrom

    @cached_property
    def utrans_dwave(self) -> Tensor:
        return self.trans_dwave * angstrom


class InterpolatedBandpass(Bandpass, InterpolatedExtinction):
    _mag_or_linear = InterpolatedExtinction._MagOrLinear.linear


@dataclass(unsafe_hash=True)
class LinearInterpolatedBandpass(InterpolatedBandpass, Linear1dInterpolated):
    _interp_data: tuple[Tensor, Tensor] = dataclasses.field(repr=False, compare=False, hash=False)
    # de-classmethod-ify...
    _interpolate = Linear1dInterpolated._interpolate.__func__
    _interp = getattr_static(Linear1dInterpolated, '_interp').__func__

    @property
    def _wave(self):
        return self._interp.x

    @property
    def _trans(self):
        return self._interp.y
