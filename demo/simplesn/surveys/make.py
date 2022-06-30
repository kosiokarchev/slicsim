import os

import numpy as np
import torch
from astropy.table import Table


SURVEY = 'pantheon-g10'
INNAME = 'Ancillary_G10.FITRES'


colnames = {
    'zCMB': 'zcmb',
    'zCMBERR': 'err_zcmb',
    'x0': 'x0', 'x0ERR': 'std_x0',
    'mB': 'm', 'mBERR': 'std_m',
    'x1': 'x1', 'x1ERR': 'std_x1',
    'c': 'c', 'cERR': 'std_c',
    'COV_x1_c': 'cov_x1_c',
    'COV_x1_x0': 'cov_x0_x1',
    'COV_c_x0': 'cov_x0_c',
}

t = (t_ := Table.read(os.path.join(SURVEY, INNAME), format='ascii.basic'))[list(colnames.keys())]
t.rename_columns(*zip(*colnames.items()))
t['cov_m_x1'] = t['cov_x0_x1'] * t['std_m'] / t['std_x0']
t['cov_m_c'] = t['cov_x0_c'] * t['std_m'] / t['std_x0']

zcmb = torch.as_tensor(t['zcmb'], dtype=torch.float32)
vars = torch.as_tensor(np.stack([
    t['std_m']**2, t['cov_m_x1'], t['cov_m_c'],
    t['cov_m_x1'], t['std_x1']**2, t['cov_x1_c'],
    t['cov_m_c'], t['cov_x1_c'], t['std_c']**2
], -1), dtype=torch.float32).reshape(-1, 3, 3)

good = (torch.linalg.eigvalsh(vars) > 0).all(-1)

print('Bad:', torch.arange(len(vars))[~good].tolist())

# torch.save(zcmb[good], os.path.join(SURVEY, f'{SURVEY}-zcmb.pt'))
# torch.save(vars[good], os.path.join(SURVEY, f'{SURVEY}-vars.pt'))
