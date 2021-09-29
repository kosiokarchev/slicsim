import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass
from inspect import signature

import forge
from feign import copy_function, feign

from .abc import Source
from ..extinction import Extinction
from ..utils import _t
from ..utils.utility_base import UtilityBase


@dataclass
class AffectedSource(Source, ABC):
    base: UtilityBase.private(Source)

    @property
    def flux_unit(self):
        return self.base.flux_unit

    def __post_init__(self):
        return
        self.__call__ = feign(copy_function(self.__call__), inspect.signature(forge.insert((
            forge.kwarg(name, default=param.default, type=param.annotation)
            for name, param in signature(self.base).parameters.items()
            if param.kind is param.KEYWORD_ONLY
        ), index=-1)(self.__call__)))

    # TODO: Serialize this!

    def __getstate__(self):
        return {'base': self.base}

    def __setstate__(self, state):
        self.base = state['base']
        self.__post_init__()

    @abstractmethod
    def flux(self, phase: _t, wave: _t, **kwargs) -> _t:
        return self.base(phase, wave, **kwargs)


@dataclass
class Phaseshifted(AffectedSource):
    phase0: _t = 0

    def flux(self, phase: _t, wave: _t, **kwargs) -> _t:
        return super().flux(phase - self.phase0, wave, **kwargs)


@dataclass
class Redshifted(AffectedSource):
    z: _t = 0

    @property
    def scale_factor(self):
        return 1 / (1+self.z)

    def flux(self, phase: _t, wave: _t, **kwargs) -> _t:
        return (a := self.scale_factor)**3 * super().flux(a*phase, a*wave, **kwargs)


class Extincted(AffectedSource):
    ext: UtilityBase.private(Extinction)

    def flux(self, phase: _t, wave: _t, **kwargs) -> _t:
        return self.ext.linear(wave) * super().flux(phase, wave, **kwargs)
