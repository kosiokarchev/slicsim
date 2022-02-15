from torch import Tensor
from typing_extensions import TypedDict

from phytorch.quantities import Quantity

from ..model import Field


class SurveyData(TypedDict, total=False):
    field: Field
    zp_flux: Quantity
    zp_mag_mean: Tensor
    zp_mag_std: Tensor
    noise: Quantity
    sky: Quantity
    area: Quantity
    gain: Quantity
