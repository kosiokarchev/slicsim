from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
from operator import itemgetter
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence, Type, TypeVar, Union

import pandas as pd
import torch
from matplotlib import pyplot as plt
from typing_extensions import TypeAlias

from clipppy import Clipppy
from clipppy.commands.nre import ClipppyDataset
from clipppy.utils.messengers import CollectSitesMessenger
from clipppy.utils.plotting.nre import MultiNREPlotter, to_percentiles
from clipppy.utils.plotting.posterior import FuncPosteriorPlotter
from libsimplesn import SimpleSN


_KT = TypeVar('_KT')
_VT = TypeVar('_VT')
_MultidictT: TypeAlias = Mapping[_KT, Union['_MultidictT', _VT]]


def to_multidict(df: Union[pd.DataFrame, pd.Series]) -> _MultidictT:
    return {
        key: to_multidict(df.loc[key])
        for key in df.index.levels[0]
    } if isinstance(df.index, pd.MultiIndex) else (
        df.T if isinstance(df, pd.DataFrame) else df
    ).to_dict()


def from_multidict(d: _MultidictT, levelnames: Sequence[str], cls: Union[Type[pd.Series], Type[pd.DataFrame], Callable[[_MultidictT], Union[pd.Series, pd.DataFrame]]] = pd.Series):
    return (
        pd.concat({key: from_multidict(val, levelnames[1:], cls) for key, val in d.items()}, axis='columns', names=levelnames[:1])
        if isinstance(next(iter(d.values()), None), Mapping) else
        cls(d)
    )


def get_priors(param_names: Iterable[str], dataset: ClipppyDataset):
    with CollectSitesMessenger(*param_names) as trace:
        dataset.get_trace()
    return {name: site['fn'] for name, site in trace.items()}


class Nrepper:
    def __init__(self, nre, labels, groups=(('Om0', 'Ode0'),)):
        self.labels = labels
        self.groups = groups
        self.priors = get_priors(nre.param_names, nre.dataset.dataset)
        self.ranges = {key: (prior.support.lower_bound, prior.support.upper_bound)
                       for key, prior in self.priors.items()}

    def __call__(self, ngrid=256, ngrid_cosmo=32):
        return MultiNREPlotter(
            groups=self.groups,
            grid_sizes=defaultdict(lambda: ngrid, Om0=ngrid_cosmo, Ode0=ngrid_cosmo),
            priors=self.priors, ranges=self.ranges, labels=self.labels
        )


@dataclass
class ResultsPlotter:
    simplesn: SimpleSN
    config: Clipppy

    version: int
    version_prefix: str = 'onlycosmo-fc/step'
    nsteps: int = 9999

    ngrid_cosmo = 128

    base_logdir = Path('lightning_logs')

    @cached_property
    def logdir(self):
        return self.base_logdir / self.simplesn.basedata_prefix / self.version_prefix / f'version_{self.version}'

    @cached_property
    def checkpointdir(self):
        return self.logdir / 'checkpoints'

    @cached_property
    def plotdir(self):
        plotdir = self.logdir / 'plots'
        plotdir.mkdir(parents=True, exist_ok=True)
        return plotdir

    @cached_property
    def bounds(self):
        return self.simplesn.hdi_bounds[self.simplesn.datatype, self.simplesn.N].to_dict()

    @cached_property
    def nre(self):
        nre = self.config.lightning_nre

        nre.dataset_config.kwargs['ranges'].update({
            key: itemgetter('lower', 'upper')(val)
            for key, val in self.bounds.items()
        })
        nre.head, nre.tail = torch.load(self.checkpointdir / f'epoch=0-step={self.nsteps}.ckpt')['clipppy_nets']
        return nre.cuda().eval()

    @cached_property
    def nrepper(self):
        return Nrepper(self.nre, self.config.kwargs['defs']['labels'])

    @cached_property
    def nrep(self):
        return self.nrepper(ngrid_cosmo=self.ngrid_cosmo)

    @cached_property
    def plotter(self):
        return self.nrep.plotters[self.nrepper.groups[0]]

    def post(self, data):
        obs = {key: data[key] for key in self.nre.obs_names}
        return self.plotter.post(obs, self.nre.head, self.nrep.subtails(self.nre.tail)[self.nrepper.groups[0]])

    def perc(self, data):
        return to_percentiles(self.post(data), self.plotter.nparams)

    def plot_nre_mc(self, ssn, hdis, levels=(0.68, 0.95)):
        data = ssn.data
        hdi = hdis[ssn.N, ssn.datatype, ssn.suffix, ssn.version]

        plotter = FuncPosteriorPlotter()
        plotter.corner_names = self.nrepper.groups[0]
        plotter.labels = self.nrepper.labels
        plotter.ranges = self.nrepper.ranges

        fig, axs = plotter._corner_figure()

        axs[1, 0].tricontour(
            *hdi, levels=levels,
            colors='green', linestyles=('-', '--')
        )
        axs[0, 0].hist(hdi[0], density=True, color='green')
        axs[1, 1].hist(hdi[1], density=True, color='green')

        plotter.default_truth_color = 'k'

        plotter.corner(
            self.nrep.grids, self.post(data),
            fig=fig, truths=data, plot_dist=False,
            contour_kwargs={'colors': 'red', 'levels': levels},
            line1d_kwargs={'color': 'red'}
        )

        plt.suptitle(ssn.data_prefix)
        return fig, axs
