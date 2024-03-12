from dataclasses import dataclass
from functools import reduce
from operator import add
from typing import Mapping, Any, TypedDict, Type

import pandas as pd
import torch
from torch import Tensor

from phytorch.quantities import Quantity, GenericQuantity

from .instruments import psf_to_area
from .units import ADU, px, electrons, linpx
from ..bandpasses.bandpass import Bandpass
from ..bandpasses.magsys import MagSys
from ..model import Field


class ExtraData(TypedDict, total=False):
    Av_MW: float
    z: float
    z_cosmo: float

    distance: GenericQuantity

    mu: Any
    distmod_err: float


@dataclass
class SurveyData:
    ZPCAL = 27.5

    field: Field

    fluxcal: Tensor = None

    # FLUXCAL
    fluxcalerr: Tensor = None

    # basic_instrument
    zp_mag_mean: Tensor = None
    zp_mag_std: Tensor = None
    ccd_noise: Quantity = None  # [ADU / px], ccd signal
    sky_noise: Quantity = None  # [ADU / px], sky signal = sigma(sky)^2
    area: Quantity = None  # [px]
    gain: Quantity = None  # [e / ADU]

    @property
    def srcflux(self) -> Quantity:
        return self.fluxcal * 10**(0.4*(self.zp_mag_mean - self.ZPCAL)) * ADU

    @property
    def bgflux(self) -> Quantity:
        return reduce(add, filter(lambda arg: arg is not None, (self.ccd_noise, self.sky_noise))) * self.area  # [ADU]

    def calc_fluxcalerr(self):
        return (((self.srcflux + self.bgflux) / self.gain)**0.5).to(ADU).value * 10**(-0.4*(self.zp_mag_mean-self.ZPCAL))

    # def src_fluxcalerr(self):
    #
    # def bg(self, zp=27.5):
    #     # SNANA manual, 4.14, (10)
    #     return ((self.background / self.gain)**0.5).to(ADU).value  # * 10**(0.4*(zp-self.zp_mag_mean))

    @classmethod
    def from_phot(cls, phot: pd.DataFrame, meta: Mapping[str, Any], bandmap: Mapping[str, Bandpass], magsys: MagSys):
        extra_data = dict(
            (name_out, val * unit if unit is not False else val)
            for name_in, name_out, unit in (
                ('MJD', 'mjd',  False),
                ('FLUXCAL', 'fluxcal', False),
                ('FLUXCALERR', 'fluxcalerr', False),
                ('ZPTAVG', 'zp_mag_mean', False),
                ('ZPTSIG', 'zp_mag_std', False),
                ('CCD_NOISE', 'ccd_sig', (electrons / px)*0.5),
                ('SKYSIG', 'sky_sig', (ADU / px)**0.5),
                ('GAIN', 'gain', electrons / ADU), ('CCD_GAIN', 'gain', electrons / ADU),
                ('PSF', 'psf', linpx), ('PSF1', 'psf', linpx),
                ('PSF2', 'psf2', linpx),
                ('PSFRATIO', 'ratio', False)
            )
            if name_in in phot
            for val in [torch.tensor(phot[name_in].to_numpy().astype(float), dtype=torch.get_default_dtype())]
        )

        if 'psf' in extra_data:
            extra_data['area'] = psf_to_area(extra_data.pop('psf'), extra_data.pop('psf2', None), extra_data.pop('ratio', 0))

        if 'ccd_sig' in extra_data:
            extra_data['ccd_noise'] = extra_data.pop('ccd_sig')**2 / extra_data['gain']
        if 'sky_sig' in extra_data:
            extra_data['sky_noise'] = extra_data.pop('sky_sig')**2

        return cls(
            field=Field(
                times=extra_data.pop('mjd') - meta.get('PEAKMJD', 0),
                bands=[bandmap[band] for band in phot['FLT']],
                magsys=magsys
            ), **extra_data
        )
