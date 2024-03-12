import os

import numpy as np
import torch

import slicsim.data


BASEDIR = 'CSP_filter_package'


def save_bandpass(group, name, wave, trans):
    fname = os.path.join(slicsim.data.__path__[0], 'bandpasses', group, f'{group}_{name}.pt')
    if os.path.exists(fname):
        print('EXISTS', fname)
    else:
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        torch.save((*(torch.as_tensor(a, dtype=torch.get_default_dtype())
                      for a in (wave, trans)),), fname)
        print(fname)


for fname in sorted(os.listdir(BASEDIR)):
    save_bandpass('cspk17', fname.split('.', 1)[0], *np.loadtxt(os.path.join(BASEDIR, fname)).T)
