from math import log, pi

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



def basic_instrument(model_flux: Quantity, zp_flux: Quantity, zp_mag: _t, noise: Quantity, gain: Quantity):
    """

    Parameters
    ----------
    model_flux:
        [same as zp_flux]
    zp_flux:
        [same as model_flux]
    zp_mag:
        [dimensionless (magnitudes)]
    noise:
        [ADU]
    gain:
        [ADU / electrons]

    Returns
    -------
        [electrons]
    """
    signal = (model_flux / zp_flux * 10**(0.4 * zp_mag) * ADU).to(ADU)
    return ((signal + noise) / gain).to(electrons).value
