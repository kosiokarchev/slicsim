from typing import Callable, ClassVar, Mapping, TYPE_CHECKING

import torch
from torch import Size, Tensor

from phytorch.interpolate import interp2d
from phytorch.interpolate.splines import SplineNd
from phytorchx import broadcast_cat

from .hsiao import HsiaoSource
from ..utils import _t, cached_property, DataRegistry, Delayed
from ..utils.utility_base import UtilityBase


class BayeSNSource(HsiaoSource):
    bayesn_phase: ClassVar[Tensor] = torch.tensor([-10, 0, 10, 20, 30, 40])  # days
    bayesn_wave: ClassVar[Tensor] = (  # angstrom
        torch.tensor([0.3, 0.43, 0.49, 0.54, 0.62, 0.77, 0.87, 1.04, 1.24, 1.65, 1.85]) * 1e4
    )

    def sed(self, phase, wave, **kwargs) -> Tensor:
        return interp2d(phase, wave, self.grid_flux,
                        self.grid_phase[(0, -1),], self.grid_wave[(0, -1),],
                        mode='bicubic', align_corners=True)

    @classmethod
    @cached_property
    def bayesn_spline(cls) -> SplineNd:
        return SplineNd(cls.bayesn_phase, cls.bayesn_wave)


    if TYPE_CHECKING:
        bayesn_grid_shape: ClassVar[Size]
        bayesn_ngrid: ClassVar[int]
    else:
        @classmethod
        @cached_property
        def bayesn_grid_shape(cls):
            return Size([cls.bayesn_phase.shape[-1], cls.bayesn_wave.shape[-1]])

        @classmethod
        @cached_property
        def bayesn_ngrid(cls):
            return cls.bayesn_grid_shape.numel()


    A: UtilityBase.private(_t)

    M0 = -19.5

    W0: Tensor
    W1: Tensor
    L: Tensor

    delta_M: _t = 0.
    theta: _t = 0.
    e: Tensor

    E: UtilityBase.private(Tensor)
    _E_shape = None

    def set_params(self, **kwargs):
        super().set_params(**kwargs)

        self.E = (self.L @ self.e.unsqueeze(-1)).squeeze(-1).unflatten(-1, self._E_shape or self.bayesn_grid_shape)

    def flux(self, phase: Tensor, wave: Tensor, **kwargs):
        theta = self.theta[..., None, None] if torch.is_tensor(self.theta) else self.theta
        # eq. (12)
        return super().flux(phase, wave) * 10 ** (-0.4 * (
            self.M0 + self.delta_M
            + self.bayesn_spline.evaluate(
                self.W0 + theta * self.W1 + self.E,
                phase.clamp(*self.bayesn_phase[(0, -1),]),
                wave.clamp(*self.bayesn_wave[(0, -1),]),
            )
            # torch.where(
            #     torch.logical_and(
            #         torch.logical_and(phase >= self.bayesn_phase[0],
            #                           phase <= self.bayesn_phase[-1]),
            #
            #         torch.logical_and(wave >= self.bayesn_wave[0],
            #                           wave <= self.bayesn_wave[-1]),
            #     ),
            #     self.bayesn_spline.evaluate(y, phase, wave),
            #     0.
            # )
        ))


class TrainedBayeSNSource(BayeSNSource, Delayed):
    # TODO: multiple delayed data sources
    _bayesn_data_func: ClassVar[Callable[[], Mapping]]

    @classmethod
    @cached_property
    def _delayed_data(cls) -> Mapping:
        return dict(enumerate(cls._delayed_data_func()), **cls._bayesn_data_func())

    bayesn_phase: ClassVar[Tensor] = Delayed.attribute('bayesn_phase', Tensor)
    bayesn_wave: ClassVar[Tensor] = Delayed.attribute('bayesn_wave', Tensor)

    W0: ClassVar[Tensor] = Delayed.attribute('W0', Tensor)
    W1: ClassVar[Tensor] = Delayed.attribute('W1', Tensor)
    L: ClassVar[Tensor] = Delayed.attribute('L', Tensor)

    def set_params(self, **kwargs):
        super().set_params(**kwargs)

        _E = self.E.new_zeros(self.bayesn_grid_shape[0], 1)
        self.E = broadcast_cat((_E, self.E, _E))


class BayeSNM20Source(TrainedBayeSNSource):
    _bayesn_data_func = DataRegistry.bayesn_M20

    _E_shape = torch.Size((6, 7))


class BayeSNT21Source(TrainedBayeSNSource):
    _bayesn_data_func = DataRegistry.bayesn_T21

    _E_shape = torch.Size((6, 4))


class BayeSNW22Source(TrainedBayeSNSource):
    _bayesn_data_func = DataRegistry.bayesn_W22

    _E_shape = torch.Size((6, 9))
