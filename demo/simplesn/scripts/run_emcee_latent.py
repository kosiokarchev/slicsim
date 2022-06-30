import sys

import argparse
import sys

import pyro
import torch
from tqdm.auto import trange

sys.path.append('.')

from libsimplesn import eye_like, SimpleSN

parser = argparse.ArgumentParser()
parser.add_argument('--survey', default='pantheon-g10')
parser.add_argument('--N', type=int)
parser.add_argument('--datatype', choices=('mphotoz', 'specz'))
parser.add_argument('--suffix', default=0)
parser.add_argument('--version', type=int)

parser.add_argument('--cpu', action='store_true')

args = vars(parser.parse_args(sys.argv[1:]))


if not args.pop('cpu', False):
    print('Running on cuda')

    # noinspection PyUnresolvedReferences
    from clipppy.patches import torch_numpy

    torch.set_default_tensor_type(torch.cuda.FloatTensor)


simplesn = SimpleSN(**args)
config = simplesn.config('simplesn-marginal.yaml', gen=False)
mc = simplesn.emcee_result.to_dataset().stack(sample=('chain', 'draw'))


loc = 0
var = 0
for i in trange(len(mc.coords['sample'])):
    res = mc.isel(sample=i)

    params = {key: torch.tensor(float(val)) for key, val in res.data_vars.items()}
    trace = pyro.condition(config.mock, data=params)(initting=False, conditioning=True)

    priormean = trace.nodes['data']['fn'].base_dist.loc
    priorvars = trace.nodes['priorvars']['value']
    likevars = trace.nodes['likevars']['value']
    obs = trace.nodes['data']['value']

    # priorvars[..., 0, 0] -= trace.nodes['R_z']['value']**2
    # likevars[..., 0, 0] += trace.nodes['R_z']['value']**2

    priorvarsinv = priorvars.inverse()
    likevarsinv = likevars.inverse()

    a = torch.stack((torch.ones_like(params['alpha']), params['alpha'], - params['beta']), -1).unsqueeze(-2)
    A = torch.cat((a, eye_like(a)), -2)

    postvars = (priorvarsinv + likevarsinv).inverse()
    postloc = postvars @ (priorvarsinv @ priormean.unsqueeze(-1) + likevarsinv @ obs.unsqueeze(-1))

    loc += (A @ postloc).squeeze(-1)
    var += A @ postvars @ A.T

loc /= len(mc.coords['sample'])
var /= len(mc.coords['sample'])

simplesn.emcee_latent = (loc, var)
