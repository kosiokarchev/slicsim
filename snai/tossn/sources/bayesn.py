from typing import ClassVar

import torch
from torch import Size, Tensor

from pyrofit.utils.interpolate import LinearNDGridInterpolator
from .hsiao import HsiaoSource
from ..extinction import Fitzpatrick99


class BayeSN(HsiaoSource):
    bayesn_phase: ClassVar[Tensor] = torch.tensor([-10, 0, 10, 20, 30, 40])  # days
    # bayesn_phase: ClassVar[Tensor] = torch.linspace(-10, 40, 51)  # days
    bayesn_wave: ClassVar[Tensor] = torch.tensor([0.3, 0.43, 0.54, 0.62, 0.77, 1.04, 1.2, 1.65, 1.85]) * 1e4  # angstrom

    bayesn_grid_shape: ClassVar[Size] = Size([bayesn_phase.shape[-1], bayesn_wave.shape[-1]])
    bayesn_ngrid: ClassVar[int] = bayesn_grid_shape.numel()

    M0 = -19.5
    Rv: Tensor
    Av: Tensor
    E: Tensor
    delta_M: Tensor
    theta: Tensor
    W: Tensor
    W0: Tensor

    def set_params(self, **kwargs):
        super().set_params(**kwargs)
        self.E = self.E.unflatten(-1, self.bayesn_grid_shape)
        self.W = self.W.unflatten(-1, self.bayesn_grid_shape)
        self.W0 = self.W0.unflatten(-1, self.bayesn_grid_shape)

    def flux(self, phase: Tensor, wave: Tensor, **kwargs):
        # eq. (12)
        return super().flux(phase, wave) * 10 ** (-0.4 * (
            self.M0
            + self.delta_M.unsqueeze(-1)
            + (self.theta.unsqueeze(-1)
               * self._interpolate(
                    LinearNDGridInterpolator((self.bayesn_phase, self.bayesn_wave), self.W),
                    phase, wave)
               ).sum(-2)
            + self._interpolate(
                LinearNDGridInterpolator((self.bayesn_phase, self.bayesn_wave), self.W0),
                phase, wave)
            + self._interpolate(
                LinearNDGridInterpolator((self.bayesn_phase, self.bayesn_wave), self.E),
                phase, wave)
            + self.Av.unsqueeze(-1) * Fitzpatrick99.mag(wave, Rv=self.Rv)
        ))