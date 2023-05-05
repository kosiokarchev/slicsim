import importlib.resources
import pathlib
import typing as tp
from functools import partial
from inspect import getattr_static

import torch
from torch import Tensor

from .. import __package__ as __root_package__


# TODO: Do this properly
datadir: pathlib.Path = importlib.resources.files(__root_package__) / 'data'


_T = tp.TypeVar('_T')


# TODO: CHECK THIS!!!!
class cached_property(property):
    def __get__(self, instance=None, typ=None):
        res = super().__get__(instance, typ)
        setattr(instance or typ, self.fget.__name__, res)
        return res

    def __set__(self, instance, value):
        instance.__dict__[self.fget.__name__] = value


def _fullpath(path):
    return datadir / path


_torch_load = partial(torch.load, map_location=torch._C._get_default_device())


def _regitem(func, *args, _propertizer=property, **kwargs):
    return staticmethod(partial(func, *args, **kwargs))


def _regtorch(path, **kwargs):
    return _regitem(_torch_load, _fullpath(path), **kwargs)


class DataRegistry:
    hsiao = _regtorch('sources/Hsiao.pt')

    bayesn_m20 = _regtorch('sources/BayeSN-M20.pt')
    bayesn_t21 = _regtorch('sources/BayeSN-T21.pt')
    bayesn_w22 = _regtorch('sources/BayeSN-W22.pt')

    salt2_4 = _regtorch('sources/SALT2-4.pt')
    extinction_salt2_4 = _regtorch('colourlaws/SALT2-4.pt')
    salt3_k21 = _regtorch('sources/SALT3-K21.pt')
    extinction_salt3_k21 = _regtorch('colourlaws/SALT3-K21.pt')

    snemo_2 = _regtorch('sources/SNEMO-2.pt')
    snemo_7 = _regtorch('sources/SNEMO-7.pt')
    snemo_15 = _regtorch('sources/SNEMO-15.pt')

    extinction_FM07 = _regtorch('colourlaws/FM07.pt')
    extinction_F99 = _regtorch('colourlaws/F99.pt')

    magsys_Vega = _regtorch('magsys/Vega.pt')


class Delayed:
    _delayed_data_func: tp.ClassVar[tp.Optional[tp.Callable[[], tp.Mapping]]] = None

    @classmethod
    @cached_property
    def _delayed_data(cls) -> tp.Mapping:
        return cls._delayed_data_func()

    _nokey = object()

    class DelayedAnnotated(cached_property):
        def _get(self, slf: 'Delayed'):
            ret = slf._delayed_data
            return ret if self.key is slf._nokey else ret[self.key]

        def __init__(self, key):
            self.key = key
            super().__init__(self._get)

    # noinspection PyUnusedLocal
    @classmethod
    def attribute(cls, key=_nokey, typ: tp.Type[_T] = None) -> _T:
        return classmethod(cls.DelayedAnnotated(key=key))


_t = tp.Union[float, Tensor]
