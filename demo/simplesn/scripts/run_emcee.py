import argparse
import os
import sys

import arviz as az
import numpy as np
import pandas as pd
import seaborn
import torch
from corner import corner
from matplotlib import pyplot as plt

from clipppy.commands.emcee import Emcee
from clipppy.utils import to_tensor, torch_get_default_device


sys.path.append('.')
from libsimplesn import SimpleSN


seaborn.reset_orig()


parser = argparse.ArgumentParser()
parser.add_argument('--survey', default='pantheon-g10')
parser.add_argument('--N', type=int)
parser.add_argument('--datatype', choices=('photoz', 'mphotoz', 'specz'))
parser.add_argument('--suffix', default=0)
parser.add_argument('--version', type=int)

parser.add_argument('--nwalkers', default=50)
parser.add_argument('--nsteps', default=1000)
parser.add_argument('--nburnin', default=200)

parser.add_argument('--cpu', action='store_true')

args = vars(parser.parse_args(sys.argv[1:]))


if not args.pop('cpu', False):
    print('Running on cuda')

    # noinspection PyUnresolvedReferences
    from clipppy.patches import torch_numpy

    torch.set_default_tensor_type(torch.cuda.FloatTensor)


nwalkers, nsteps, nburnin = map(args.pop, ('nwalkers', 'nsteps', 'nburnin'))


simplesn = SimpleSN(**args)
config = simplesn.config('simplesn-marginal.yaml', gen=False)
truths = torch.load(simplesn.data_name, map_location=torch_get_default_device())

print('Analysing', simplesn.data_name)

mc = Emcee(config, nwalkers, exclude=('z',))
_initial_state = {
    key: to_tensor(val['value']).expand(mc.nwalkers)
    for key, val in config.mock(
        conditioning=True, initting=True, plate_stack=[mc.nwalkers]
    ).nodes.items()
    if key in mc.param_sites
}
initial_state = np.random.normal(pd.DataFrame(mc.deconstrain(_initial_state)).to_numpy(), 0.2)

mc.run_mcmc(initial_state, nsteps, progress=True)

print('Saving', simplesn.emcee_result_name)
simplesn.emcee_result = mc

idata = az.convert_to_inference_data(mc.constrain(mc.get_batched_params(mc.chain)))['posterior']

os.makedirs(simplesn.plotdir, exist_ok=True)

print('Plotting trace')
az.plot_trace(idata, lines=[(key, {}, (float(truths[key]),)) for key in idata.keys()],
              coords={'draw': slice(0, None)})
plt.savefig(simplesn.plotdir / f'{simplesn.emcee_prefix}-trace.png')
plt.close()

print('Plotting corner')
corner(idata[{'draw': slice(nburnin, None)}], levels=(0.68, 0.95), truths=truths)
plt.savefig(simplesn.plotdir / f'{simplesn.emcee_prefix}-corner.png')
plt.close()
