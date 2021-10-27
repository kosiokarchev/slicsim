from collections import Mapping, OrderedDict
from dataclasses import dataclass
from functools import partial
from itertools import chain
from random import sample
from typing import Iterable, Union

import torch
from frozendict import frozendict
from torch import Tensor
from torch.nn import LazyLinear
from torch.utils.data import IterableDataset

from clipppy import Clipppy
from clipppy.distributions.conundis import ConstrainingMessenger


def sample_all(arr):
    return sample(arr, len(arr) // 2)


@dataclass
class CPDataset(IterableDataset[tuple[Mapping[str, Tensor], Mapping[str, Tensor]]]):
    config: Clipppy
    param_names: Iterable[str]
    data_names: Iterable[str]
    ranges: Mapping[str, tuple[Union[float, Tensor, None], Union[float, Tensor, None]]]

    @property
    def constraining_messenger(self):
        return ConstrainingMessenger(self.ranges)

    def __next__(self):
        while True:
            try:
                with torch.no_grad(), self.constraining_messenger:
                    mock = self.config.mock()
                return tuple(
                    OrderedDict((name, mock.nodes[name]['value']) for name in names)
                    for names in (self.param_names, self.data_names)
                )
            except ValueError as e:
                pass

    def __iter__(self):
        return self


class FilesDataset(IterableDataset):
    def __init__(self, fnames, load_kwargs=frozendict()):
        self.fnames = fnames
        self.load = partial(torch.load, **load_kwargs)

    def iter(self):
        while True:
            yield from chain.from_iterable(
                map(sample_all, map(self.load, sample_all(self.fnames))))

    def __iter__(self):
        return self.iter()


class Head(torch.nn.Module):
    def __init__(self, net: torch.nn.Module):
        super().__init__()
        self.net = net

    def preprocess(self, params: Mapping[str, Tensor], obs: Mapping[str, Tensor]):
        return tuple(
            torch.stack(tuple(val.values()), -1).squeeze(-1)
            for val in (params, obs)
        )

    def forward(self, params: Mapping[str, Tensor], obs: Mapping[str, Tensor]):
        theta, x = self.preprocess(params, obs)
        return theta, self.net(x)


class NRE(torch.nn.Module):
    def __init__(self, net: torch.nn.Module, features=256):
        super().__init__()
        self.l_theta = LazyLinear(features)
        self.l_x = LazyLinear(features)
        self.net = net

    def forward(self, theta: Tensor, x: Tensor):
        return self.net(self.l_theta(theta) + self.l_x(x))
