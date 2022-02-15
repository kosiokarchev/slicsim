from math import log, pi

from .units import ADU, electrons


def fwhm_to_area(fwhm):
    return pi * (fwhm/2)**2 / log(2)


def basic_instrument(model_flux, zp_flux, zp_mag, noise, gain):
    signal = (model_flux / zp_flux * 10**(0.4 * zp_mag) * ADU).to(ADU)
    return ((signal + noise) / gain).to(electrons).value
