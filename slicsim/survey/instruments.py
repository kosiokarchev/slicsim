from math import log, pi

from torch import Tensor

from phytorch.quantities import Quantity

from .units import ADU, electrons
from ..utils import _t


def fwhm_to_area(fwhm):
    return pi * (fwhm/2)**2 / log(2)


def psf_to_area(psf1: Quantity, psf2: Quantity = None, ratio: _t = 0):
    if psf2 is None:
        psf2 = psf1 * 0  # preserves units

    a2, b2 = psf1**2, psf2**2
    a2pb2 = a2 + b2
    return 4*pi * a2pb2 * (a2 + ratio * b2)**2 / ((ratio**2 * b2 + a2) * a2pb2 + 4*ratio*a2*b2)



def basic_instrument(countscal: Tensor, zp_mag: _t, background: Quantity, gain: Quantity):
    """

    Parameters
    ----------
    countscal:
        [dimensionless]
    zp_mag:
        [dimensionless (magnitudes)]
    background:
        [ADU]
    gain:
        [electrons / ADU]

    Returns
    -------
        [electrons]
    """
    signal = countscal * 10**(0.4 * zp_mag) * ADU
    return ((signal + background) * gain).to(electrons).value
