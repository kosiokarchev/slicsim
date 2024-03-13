# SLICsim: A Supernova LIght-Curve simulator for the machine learning age

## Installation

Installation from source can be done with `pip install .` from the root directory of the repo. This will also attempt to install the few dependencies ([`torch`](https://pytorch.org/), [`torch-scatter`](https://github.com/rusty1s/pytorch_scatter), and [`phytorch`](https://github.com/kosiokarchev/phytorch)).

An official release will be drawn shortly and published on PyPI.

## Example usage

To generate a multi-band SN Ia light curve using the BayeSN model (and default parameters):
```python
import random

import torch

from slicsim.bandpasses import des_u, des_g, des_r, des_i
from slicsim.bandpasses.magsys import AB
from slicsim.model import Field, LightcurveModel
from slicsim.sources.bayesn import BayeSNM20Source as BayeSN
from slicsim.sources.effects import Distance


model = LightcurveModel(
    Distance(BayeSN()),  # default distance is 10pc
    Field(
        # 100 random times within [-10, 40] days
        times=-10 + 50*torch.rand(100),
        # a random band for each observation
        bands=random.choices((des_u, des_g, des_r, des_i), k=100),
        magsys=AB
    )
)

fluxcal = model.bandcountscal()  # a torch.Tensor of shape (100,)
# You can use `model.field.band_indices` to separate observations
# by band and e.g. plot them:
```

![Example light curve](https://i.imgur.com/8kCoKhk.png "")

More examples and full documentation are coming soon!

## Citation

This software was presented in [SIDE-real: Truncated marginal neural ratio estimation for Supernova Ia Dust Extinction with real data](https://arxiv.org/abs/2403.07871), currently under review. If you use it, please cite
```bibtex
@article{Karchev-sidereal,
    title={
        SIDE-real: Truncated marginal neural ratio estimation
        for Supernova Ia Dust Extinction with real data},
    author={
        Karchev, Konstantin and
        Grayling, Matthew and
        Boyd, Benjamin M. and
        Trotta, Roberto and
        Mandel, Kaisey S. and
        Weniger, Christoph},
    year={2024}, month=mar,
    publisher={arXiv}, number={arXiv:2403.07871},
    url={http://arxiv.org/abs/2403.07871},
    note={arXiv:2403.07871 [astro-ph]}
}
```
