import numpy as np
import torch
import extinction


wave = torch.linspace(1000, 20000, 191)
Rvs = torch.linspace(0.1, 10, 91)

torch.save(
    ((wave, Rvs), torch.tensor(np.stack([
        extinction.fitzpatrick99(wave.numpy().astype(float), a_v=1., r_v=Rv)
        for Rv in Rvs.tolist()
    ]), dtype=torch.float32).T.contiguous()),
    '../slicsim/data/colourlaws/F99.pt'
)
