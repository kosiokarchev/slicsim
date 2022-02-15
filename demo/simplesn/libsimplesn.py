from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal, Union

import torch
from more_itertools import lstrip
from torch import Size, Tensor

from phytorch.cosmology.core import H100

from clipppy import load_config
from clipppy.yaml.jinja import parse


def eye_like(a: Tensor):
    return torch.eye(a.shape[-1], dtype=a.dtype, device=a.device)


def fancy_align(*tensors: Tensor):
    shapes = dict(enumerate(
        Size(lstrip(t.shape, lambda s: s == 1))
        for t in tensors
    ))

    shape = Size()
    js = {}
    for i, tshape in sorted(shapes.items(), key=lambda arg: len(arg[1])):
        j = 0
        for j in range(len(tshape)+1):
            try:
                shape = torch.broadcast_shapes(shape, tshape[:len(tshape)-j])
                break
            except RuntimeError:
                pass
        js[i] = j
    maxj = max(js.values())
    return tuple(
        t.reshape(t.shape + (maxj - js[i])*(1,))
        for i, t in enumerate(tensors)
    )


def phillips(M0, x1, c, alpha, beta):
    M0, x1, c, alpha, beta = fancy_align(M0, x1, c, alpha, beta)
    return M0 + alpha * x1 + beta * c


def combine_noise_and_scatter(scale_tril: Tensor, sigma_res: Tensor):
    V = scale_tril @ scale_tril.transpose(-2, -1)
    V.diagonal(0, -2, -1)[::3] = V.diagonal(0, -2, -1)[::3] + (
        torch.tensor(sigma_res, dtype=scale_tril.dtype, device=scale_tril.device)
        if not torch.is_tensor(sigma_res) else sigma_res
    )[..., None, None]**2
    return V


def pack_data(m, x1, c):
    return torch.stack((m, x1, c), -1).flatten(start_dim=-2)


@dataclass
class SimpleSN:
    survey: str
    N: int = None
    suffix: Union[str, Any] = ''

    datatype: str = ''
    vitype: Literal['mvn', 'hpmvn'] = 'hpmvn'
    nretype: str = ''

    _data: ClassVar[Path] = Path('data')
    _res: ClassVar[Path] = Path('res')
    _survey: ClassVar[Path] = Path('surveys')

    @property
    def _surveydir(self):
        return self._survey / self.survey

    @property
    def prefix(self):
        return '-'.join(filter(bool, map(str, (self.survey, self.N, self.suffix))))

    @property
    def data_prefix(self):
        return '-'.join(filter(bool, (self.prefix, self.datatype)))

    @property
    def datadir(self):
        return self._data / self.survey / self.prefix

    @property
    def data_name(self):
        return self.datadir / f'{self.data_prefix}-data.pt'

    @property
    def zcmb_name(self):
        return self.datadir / f'{self.prefix}-zcmb.pt'

    @property
    def scale_tril_name(self):
        return self.datadir / f'{self.prefix}-scale_tril.pt'

    @property
    def resdir(self):
        return self._res / self.survey / self.prefix

    @property
    def _vi_prefix(self):
        return '-'.join(filter(bool, (self.data_prefix, 'vi', self.vitype)))

    @property
    def vi_losses_name(self):
        return self.resdir / f'{self._vi_prefix}-losses.pt'

    @property
    def vi_losses(self):
        return torch.load(self.vi_losses_name)

    @vi_losses.setter
    def vi_losses(self, value):
        torch.save(value, self.vi_losses_name)

    @property
    def vi_guide_name(self):
        return self.resdir / f'{self._vi_prefix}-guide.pt'

    @property
    def vi_guide(self):
        return torch.load(self.vi_guide_name)

    @vi_guide.setter
    def vi_guide(self, value):
        torch.save(value, self.vi_guide_name)

    @property
    def _nre_prefix(self):
        return '-'.join(filter(bool, (self.data_prefix, 'nre', self.nretype)))

    @property
    def nre_losses_name(self):
        return self.resdir / f'{self._vi_prefix}-losses.pt'

    @property
    def nre_losses(self):
        return torch.load(self.nre_losses_name)

    @nre_losses.setter
    def nre_losses(self, value):
        torch.save(value, self.nre_losses_name)

    @property
    def nre_nets_name(self):
        return self.resdir / f'{self._nre_prefix}-nets.pt'

    @property
    def nre_nets(self):
        return torch.load(self.nre_nets_name)

    @nre_nets.setter
    def nre_nets(self, value):
        torch.save(value, self.nre_nets_name)

    @property
    def plotdir(self):
        return self.resdir / 'plots'

    def config(self, fname='simplesn.yaml', gen=False, **kwargs):
        config = load_config(parse(
            fname,
            zcmb_name=self.zcmb_name,
            scale_tril_name=self.scale_tril_name,
            data_name=self.data_name,
            gen=gen, **kwargs
        ))
        config.kwargs['defs']['cosmo'].obj.H0 = H100
        return config
