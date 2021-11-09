from abc import ABC, abstractmethod
from typing import Callable, ClassVar, Tuple

from phytorch.interpolate import LinearNDGridInterpolator
from phytorch.interpolate.abc import AbstractNDInterpolator
from phytorch.units.cgs import erg
from phytorch.units.si import angstrom, second
from phytorch.units.unit import Unit
from torch import Tensor

from ..utils import _t, cached_property, Delayed
from ..utils.utility_base import UtilityBase


class Source(UtilityBase, ABC):
    @abstractmethod
    def flux(self, phase: _t, wave: _t, **kwargs) -> _t: ...

    flux_unit: ClassVar[Unit] = erg / second / angstrom

    def __call__(self, phase: _t, wave: _t, **kwargs):
        self.set_params(**kwargs)
        return self.flux(phase, wave, **kwargs)


class ColouredSource(Source, ABC):
    coeff_colour: _t

    @abstractmethod
    def colourlaw(self, phase: _t, wave: _t) -> _t: ...  # TODO: document linear!

    def flux(self, phase: _t, wave: _t, **kwargs) -> _t:
        return super().flux(phase, wave, **kwargs) * self.colourlaw(phase, wave)**self.coeff_colour


class AbstractInterpSource(Source, ABC):
    @staticmethod
    def _interpolate(ipol: AbstractNDInterpolator, phase, wave):
        return ipol(ipol.interp_input(phase, wave))


class GridInterpSource(AbstractInterpSource, ABC):
    grid_phase: Tensor  # [N_phase]
    grid_wave:  Tensor  # [N_wave]
    grid_flux:  Tensor  # [..., N_phase, N_wave]

    @property
    def grid_interpolator(self) -> LinearNDGridInterpolator:
        return LinearNDGridInterpolator((self.grid_phase, self.grid_wave), self.grid_flux)

    def interpolate_flux(self, phase: _t, wave: _t) -> Tensor:
        return self._interpolate(self.grid_interpolator, phase, wave)


class TrainedGridInterpSource(GridInterpSource, ABC):
    grid_phase: ClassVar[Tensor]  # [N_phase]
    grid_wave:  ClassVar[Tensor]  # [N_wave]
    grid_flux:  ClassVar[Tensor]  # [..., N_phase, N_wave]

    @classmethod
    @cached_property
    def grid_interpolator(cls) -> LinearNDGridInterpolator:
        return classmethod(super().grid_interpolator.fget).__get__(cls, cls)()


class DelayedGridInterpSource(TrainedGridInterpSource, Delayed, ABC):
    _delayed_data_func: ClassVar[Callable[[], Tuple[Tensor, Tensor, Tensor]]]
    grid_phase = Delayed.attribute(0, Tensor)
    grid_wave = Delayed.attribute(1, Tensor)
    grid_flux = Delayed.attribute(2, Tensor)


class SEDSource(Source):
    coeff_0: _t = 1.

    @abstractmethod
    def sed(self, phase: _t, wave: _t, **kwargs) -> Tensor: ...

    def flux(self, phase: _t, wave: _t, **kwargs) -> _t:
        return self.coeff_0 * self.sed(phase, wave)


class GridInterpSEDSource(GridInterpSource, SEDSource):
    sed = GridInterpSource.interpolate_flux


class TrainedGridInterpSEDSource(TrainedGridInterpSource, GridInterpSEDSource):
    pass


class DelayedGridInterpSEDSource(DelayedGridInterpSource, GridInterpSEDSource):
    pass


class PCASource(Source):
    coeff_0: _t = 1.
    coeffs: _t = 0.

    @abstractmethod
    def component_fluxes(self, phase: _t, wave: _t) -> Tensor: ...

    def flux(self, phase: _t, wave: _t, **kwargs) -> _t:
        component_fluxes = self.component_fluxes(phase, wave)
        return self.coeff_0 * (component_fluxes[..., 0] + (self.coeffs * component_fluxes[..., 1:]).sum(-1))


class GridInterpPCASource(GridInterpSource, PCASource, ABC):
    def component_fluxes(self, phase: _t, wave: _t) -> Tensor:
        return self.interpolate_flux(phase, wave).movedim(-2, -1)


class TrainedGridInterpPCASource(TrainedGridInterpSource, GridInterpPCASource):
    pass


class DelayedGridInterpPCASource(DelayedGridInterpSource, GridInterpPCASource):
    pass
