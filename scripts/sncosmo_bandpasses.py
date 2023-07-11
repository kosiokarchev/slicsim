import os
from typing import Iterable, Mapping, Union

import torch

import snai.tossn.data
import sncosmo as sn


def stsci_fname(i, suffix='w'):
    return f'f{i:0>3}{suffix}'


def stsci_fnames(fmap: Union[Mapping[str, Iterable[int]], Iterable[int]]):
    return tuple(
        (stsci_fname(val, key) for key, vals in fmap.items() for val in vals)
        if isinstance(fmap, Mapping) else map(stsci_fname, fmap))


bpmap = {
    'bessell': ('ux', *'bvri'),
    'snls3_landolt': 'ubvri',
    'des': 'grizy',
    'sdss': 'ugriz',
    'acs_wfc': stsci_fnames({
        'w': (435, 475, 555, 606, 625, 775, 814), 'lp': (850,)}),
    'nic2': stsci_fnames((110, 160)),
    'wfc3_ir': stsci_fnames(
        {'w': (105, 110, 125, 140, 160), 'm': (98, 127, 139, 153)}),
    'wfc3_uvis': stsci_fnames(
        {'w': (218, 225, 275, 336, 390, 438, 475, 555, 606, 625, 775, 814,),
         'x': (300,), 'm': (689, 763, 845, ), 'lp': (350, 850)}),
    'csp': ('b', 'v3009', 'v3014', 'v9844', *map(''.join, product('hjy', 'sd')), *'gikru'),
    'nircam': stsci_fnames({
        'w': (70, 90, 115, 150, 200, 277, 356, 444),
        'm': (140, 162, 182, 210, 250, 300, 335, 360, 410, 430, 460, 480)}),
    'miri': stsci_fnames((560, 770, 1000, 1130, 1500, 1800, 2100, 2550)),
    'lsst': 'ugrizy',
    'kepler': ('',),
    'keplercam': ('us', *'bvri'),
    '4shooter2': ('us', *'bvri'),
    'roman_wfi': stsci_fnames({'': (62, 87, 106, 129, 158, 184, 213, 146)}),
    'ztf': 'gri',
    'ps1': ('open', *'grizyw'),
    'uvot': (*'buv', 'white', 'uvm2', 'uvw1', 'uvw2')
}

duplicates = stsci_fnames({'w': (475, 555, 606, 625, 775, 814), 'lp': (850,)})

for key, names in bpmap.items():
    snkey = (
        'standard::' if key == 'snls3_landolt' else
        '' if 'wfc' in key or key in ('nircam', 'miri', 'roman_wfi') else
        'nic' if key == 'nic2' else
        f'{key}::' if key in ('keplercam', '4shooter2', 'uvot', 'ps1')
        else key
    )
    for name in names:
        _snkey = 'uv' if key == 'wfc3_uvis' and name in duplicates else snkey
        try:
            bp = sn.get_bandpass(_snkey + name)
            fname = os.path.join(snai.tossn.data.__path__[0], 'bandpasses', key, f'{key}_{name}.pt')

            if os.path.exists(fname):
                print('EXISTS', fname)
            else:
                os.makedirs(os.path.dirname(fname), exist_ok=True)
                torch.save((*(torch.as_tensor(a, dtype=torch.get_default_dtype())
                              for a in (bp.wave, bp.trans)),), fname)
                print(fname)
        except Exception as e:
            print(f'{key} (-> {_snkey}), {name} unsuccessful: {e}')
