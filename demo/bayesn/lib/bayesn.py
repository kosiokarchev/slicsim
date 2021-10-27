import io
from typing import Container, Sequence

import pyro, pyro.distributions
import torch
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from torch import Tensor

from clipppy import Clipppy, load_config

from phytorch.units.si import second
from phytorch.cosmology.core import FLRW

from snai.tossn.bandpasses.Bandpass import Bandpass
from snai.tossn.model import LightcurveModel, Field
from snai.tossn.sources.bayesn import BayeSN

from .nn import CPDataset


def get_scale_tril(scale, tril):
    return scale.unsqueeze(-1) * tril


class Model(LightcurveModel):
    def __init__(self, source, times: Tensor, filters: Sequence[Bandpass], **kwargs):
        super().__init__(
            source=source,
            field=Field(
                times=torch.cat(len(filters)*(times,)),
                bands=sum((len(times)*[filt] for filt in filters), [])),
            **kwargs)

    def __call__(self, cosmo: FLRW, **cosmo_kwargs):
        for key, val in cosmo_kwargs.items():
            setattr(cosmo, key, val)
        return -2.5 * self.bandcounts().to(1/second).value.log10()


def load(name, data_gen, N, gen=False, ntimes=51, guidename=None) -> Clipppy:
    config = load_config(io.StringIO(Environment(
        loader=FileSystemLoader("config/"),
        undefined=StrictUndefined
    ).from_string(open(f'config/{name}.yaml').read()).render(
        N=N, ngrid=BayeSN.bayesn_ngrid, Ntimes=ntimes,
        mockname=f'data/data-{data_gen}.pt',
        gen=gen
    )))

    if guidename and not gen:
        config.guide = torch.load(guidename)

    return config


def get_mock_trace(config):
    return config.kwargs['defs']['data']


def get_truths(config: Clipppy):
    return {
        key: val['value']
        for key, val in get_mock_trace(config).nodes.items()
    }


def get_ranges(config: Clipppy, ignore: Container[str] = (), nsamples=1000, nsigma=3):
    with pyro.plate('plate', nsamples):
        return {
            key: (m - nsigma*s, m + nsigma*s)
            for key, val in config.guide().items()
            if key not in ignore
            for m, s in [(val.mean(0), val.std(0))]
        }


CONDITION_GROUPS = {
    'E': ('E/scale', 'E/tril', 'E'),
    'W': ('W', 'W0')
}

CONDITION_REF = {
    '': ('z_cosmo',) + CONDITION_GROUPS['W'] + CONDITION_GROUPS['E']
}


def get_conditioning(config, key):
    truths = get_truths(config)
    return {key: truths[key] for key in CONDITION_REF[key]}


def get_cpdataset(config, key, ranges, prior_extent):
    config.conditioning = get_conditioning(config, key)

    config.mock.savename = None
    config.mock.conditioning = True
    config.mock.initting = False

    return CPDataset(config, ('Om0', 'Ode0'), ('obs',), {
        **ranges, 'Om0': prior_extent[:2], 'Ode0': prior_extent[2:]})