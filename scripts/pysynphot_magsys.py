from pathlib import Path

import torch
from astropy.io import fits

import slicsim.data


BASEDIR = Path(slicsim.data.__path__[0]) / 'magsys'

for fname in Path('/home/kosio/Projects/Scientific/snai/bayesn-public/grp/redcat/trds/calspec').glob('alpha_lyr_*.fits'):
    data = fits.open(fname)[1].data
    data = tuple(map(lambda arr: torch.tensor(arr.tolist()), (data['WAVELENGTH'], data['FLUX'])))

    oname = (BASEDIR / fname.stem).with_suffix('.pt')
    torch.save(data, oname)
    print(oname)
