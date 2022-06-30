from __future__ import annotations

import os
from dataclasses import dataclass, replace
from math import log
from pathlib import Path
from typing import Any, ClassVar, Literal, Union

import torch
from more_itertools import lstrip
from torch import Size, Tensor

from clipppy import load_config
from clipppy.utils import to_tensor, torch_get_default_device
from clipppy.yaml.jinja import parse
from phytorch.cosmology.core import FLRW, H100
from phytorch.math import conjugate, cosh, realise, sinc


def eye_like(a: Tensor):
    return torch.eye(a.shape[-1], dtype=a.dtype, device=a.device)


def fancy_align(*tensors: Tensor):
    tensors = tuple(map(to_tensor, tensors))
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
    return M0 - alpha * x1 + beta * c


def d_dz(out, z):
    return torch.autograd.grad(out, z, torch.ones_like(out), retain_graph=False)[0]


class GradCosmo(FLRW):
    def d_transform_curved(self, distance_dimless):
        s = sinc(self.isqrtOk0_pi * distance_dimless)
        return realise((conjugate(s) - s) / 2 + cosh(self.sqrtOk0 * distance_dimless))

    def ddz_comoving_trasverse_distance_dimless(self, z):
        return self.d_transform_curved(self.comoving_distance_dimless(z)) * self.inv_efunc(z)

    def ddz_luminosity_distance_dimless(self, z):
        return self.comoving_transverse_distance_dimless(z) + (z+1) * self.ddz_comoving_trasverse_distance_dimless(z)

    def ddz_distmod(self, z):
        return 5 / log(10) / self.luminosity_distance_dimless(z) * self.ddz_luminosity_distance_dimless(z)


def marginal_variance(alpha, beta, sigma_res=0., R_x1=0., R_c=0., R_z=None):
    has_R_z = R_z is not None

    if has_R_z:
        alpha, beta, sigma_res, R_x1, R_c, R_z = fancy_align(
            alpha, beta, sigma_res, R_x1, R_c, R_z)
    else:
        alpha, beta, sigma_res, R_x1, R_c = fancy_align(
            alpha, beta, sigma_res, R_x1, R_c)

    res = torch.diag_embed(stack_data(
        sigma_res**2 + (R_z**2 if has_R_z else 0)
        + (alpha * R_x1)**2 + (beta * R_c)**2,
        R_x1**2, R_c**2
    ))

    res[..., 0, 1] = res[..., 1, 0] = -alpha * R_x1**2
    res[..., 0, 2] = res[..., 2, 0] = beta * R_c**2

    return res if has_R_z else res.unsqueeze(-3)


def combine_noise_and_scatter(scale_tril: Tensor, sigma_res: Tensor):
    V = scale_tril @ scale_tril.transpose(-2, -1)
    V.diagonal(0, -2, -1)[::3] = V.diagonal(0, -2, -1)[::3] + (
        torch.tensor(sigma_res, dtype=scale_tril.dtype, device=scale_tril.device)
        if not torch.is_tensor(sigma_res) else sigma_res
    )[..., None, None]**2
    return V


def stack_data(m, x1, c):
    return torch.stack(torch.broadcast_tensors(*map(to_tensor, (m, x1, c))), -1)


def pack_data(m, x1, c):
    return stack_data(m, x1, c).flatten(start_dim=-2)


class file_property(property):
    owner: str
    name: str

    def __set_name__(self, owner, name):
        self.name = name + '_name'

    def __init__(self, prepare=lambda obj, value: None):
        super().__init__(self.fget, self.fset)
        self.prepare = prepare

    def fget(self, obj):
        return torch.load(getattr(obj, self.name), map_location=torch_get_default_device())

    def fset(self, obj, value):
        self.prepare(obj, value)
        path = Path(getattr(obj, self.name))
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(value, path)


def _dashed(*args):
    return '-'.join(filter(bool, map(str, args)))


@dataclass
class SimpleSN:
    survey: str
    N: int = None
    suffix: Union[str, Any] = ''
    version: Union[str, Any] = ''

    datatype: str = ''
    vitype: Literal['mvn', 'hpmvn'] = 'hpmvn'
    hmctype: Literal['nuts'] = 'nuts'
    nretype: str = ''

    _data: ClassVar[Path] = Path('data')
    _res: ClassVar[Path] = Path('res')
    _survey: ClassVar[Path] = Path('surveys')

    @property
    def _surveydir(self):
        return self._survey / self.survey

    prefix = property(lambda self: _dashed(self.survey, self.N, self.suffix))
    relpath = property(lambda self: os.path.join(self.survey, self.prefix))
    datadir = property(lambda self: self._data / self.relpath)
    resdir = property(lambda self: self._res / self.relpath)
    plotdir = property(lambda self: self.resdir / 'plots')
    hdidir = property(lambda self: self._res / 'hdi' / self.survey)
    zoomdir = property(lambda self: self._res / 'zoom')

    zcmb_name = property(lambda self: self.datadir / f'{self.prefix}-zcmb.pt')
    zcmb = file_property()

    vars_scale_tril_name = property(lambda self: self.datadir / f'{self.prefix}-vars_scale_tril.pt')
    vars_scale_tril = file_property()

    scale_tril_name = property(lambda self: self.datadir / f'{self.prefix}-scale_tril.pt')
    scale_tril = file_property()

    basedata_prefix = property(lambda self: _dashed(self.prefix, self.datatype))
    data_prefix = property(lambda self: _dashed(self.basedata_prefix, self.version))
    data_name = property(lambda self: self.datadir / f'{self.data_prefix}-data.pt')
    data = file_property()
    trace_name = property(lambda self: self.datadir / f'{self.data_prefix}-trace.pt')
    trace = file_property()

    vi_prefix = property(lambda self: _dashed(self.data_prefix, 'vi', self.vitype))
    vi_losses_name = property(lambda self: self.resdir / f'{self.vi_prefix}-losses.pt')
    vi_losses = file_property()
    vi_guide_name = property(lambda self: self.resdir / f'{self.vi_prefix}-guide.pt')
    vi_guide = file_property()

    hmc_prefix = property(lambda self: _dashed(self.data_prefix, 'hmc', self.hmctype))
    hmc_result_name = property(lambda self: self.resdir / f'{self.hmc_prefix}-result.pt')

    @file_property
    def hmc_result(self, mcmc):
        mcmc.kernel.model = mcmc.kernel.potential_fn.__self__.model = None

    nre_prefix = property(lambda self: _dashed(self.basedata_prefix, 'nre', self.nretype))
    nre_losses_name = property(lambda self: self.resdir / f'{self.nre_prefix}-losses.pt')
    nre_losses = file_property()
    nre_nets_name = property(lambda self: self.resdir / f'{self.nre_prefix}-nets.pt')
    nre_nets = file_property()

    emcee_prefix = property(lambda self: f'{self.data_prefix}-emcee')
    emcee_result_name = property(lambda self: self.resdir / f'{self.emcee_prefix}-result.pt')
    emcee_result = file_property()
    emcee_latent_name = property(lambda self: self.resdir / f'{self.emcee_prefix}-latent.pt')
    emcee_latent = file_property()

    hdi_prefix = property(lambda self: self.survey)
    hdi_mc_name = property(lambda self: self.hdidir / f'{self.hdi_prefix}-hdi.pt')
    hdi_mc = file_property()
    hdi_bounds_name = property(lambda self: self.hdidir / f'{self.hdi_prefix}-hdi-bounds.pt')
    hdi_bounds = file_property()
    hdi_nre_name = property(lambda self: self.hdidir / f'{self.hdi_prefix}-hdi-nre.pt')
    hdi_nre = file_property()

    zoom_bounds_name = property(lambda self: self.zoomdir / f'{self.data_prefix}-bounds.pt')
    zoom_bounds = file_property()
    zoom_posts_name = property(lambda self: self.zoomdir / f'{self.data_prefix}-posts.pt')
    zoom_posts = file_property()

    def config(self, fname='simplesn.yaml', gen=False, **kwargs):
        config = load_config(parse(
            fname,
            zcmb_name=self.zcmb_name,
            vars_scale_tril_name=self.vars_scale_tril_name,
            scale_tril_name=self.scale_tril_name,
            data_name=self.data_name,
            photoz='photoz' in self.datatype,
            vitype=self.vitype,
            gen=gen, **kwargs
        ))
        # TODO: init cosmo.H0
        config.kwargs['defs']['cosmo'].obj.H0 = H100
        return config

    def clone(self, **kwargs) -> SimpleSN:
        return replace(self, **kwargs)
