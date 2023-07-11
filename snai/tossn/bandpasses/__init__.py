from .bandpass import LinearInterpolatedBandpass
from ..utils import _torch_load, datadir


bandpassdir = datadir / 'bandpasses'


def __dir__():
    return [p.stem for p in bandpassdir.rglob(f'*.pt')]


def __getattr__(name):
    if name == '__all__':
        return __dir__()
    elif name.startswith('__4shooter2'):
        name = name[2:]

    try:
        globals()[name] = ret = LinearInterpolatedBandpass(
            name,
            _torch_load(next(bandpassdir.rglob(f'*/{name}.pt')))
        )
        return ret
    except StopIteration:
        raise NameError(f'No bandpass named \'{name}\'.') from None
