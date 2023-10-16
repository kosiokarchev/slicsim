from collections import defaultdict
from pathlib import Path
from typing import NamedTuple, Mapping, Iterable, TypeVar

from astropy.io import fits
from astropy.io.fits import BinTableHDU, HDUList
from astropy.table import Table

from clipppy.utils import to_tensor
from ...bandpasses.bandpass import Bandpass, LinearInterpolatedBandpass
from ...bandpasses.magsys import MagSys, CompositeMagSys, InterpolatedSpectralMagSys, PicklableMagSysMeta


class KCor(NamedTuple):
    magsys: Mapping[str, MagSys]
    zpoffs: Mapping[str, CompositeMagSys.BandZP]
    bands:  Mapping[str, Bandpass]

    @staticmethod
    def get_magsys(magsys_hdu: BinTableHDU):
        assert magsys_hdu.name == 'PrimarySED'

        t = Table(magsys_hdu.data)
        wave = to_tensor(t.columns[0].tolist())
        return {name: InterpolatedSpectralMagSys(
            name, wave,
            to_tensor(t[name].tolist()) / 10  # per 10 angstrom...
        ) for name in t.columns[1:]}

    @staticmethod
    def get_zpoffs(zpoff_hdu: BinTableHDU, magsys: Mapping[str, MagSys]):
        assert zpoff_hdu.name == 'ZPoff'

        return {
            row[0].strip(): CompositeMagSys.BandZP(magsys[row[1].strip()], row[2])
            for row in Table(zpoff_hdu.data).iterrows()
        }

    @staticmethod
    def get_bands(filter_hdu: BinTableHDU, thresh=1e-6):
        assert filter_hdu.name == 'FilterTrans'

        t = Table(filter_hdu.data)
        wave = to_tensor(t.columns[0].tolist())
        return {
            name: LinearInterpolatedBandpass(name, (
                wave[mask], trans[mask]
            )) for name in t.columns[1:]
            for trans in [to_tensor(t[name].tolist())]
            for mask in [trans >= thresh]
        }

    @classmethod
    def from_fits(cls, file: HDUList, trans_thresh=1e-6):
        return cls(
            magsys=(magsys := cls.get_magsys(file[6])),
            zpoffs=cls.get_zpoffs(file[1], magsys),
            bands=cls.get_bands(file[5], trans_thresh)
        )


_KT = TypeVar('_KT')
_VT = TypeVar('_VT')


def multivaluedict(items: Iterable[tuple[_KT, _VT]]) -> dict[_KT, list[_VT]]:
    res = defaultdict(list)
    for key, val in items:
        res[key].append(val)
    return res


def get_survey_magsys(name, kcor_names: Iterable, trans_thresh=1e-6) -> tuple[dict[tuple[str, str], list[Bandpass]], MagSys]:
    kcor_names = list(map(Path, kcor_names))

    kcors = {kcor_name.stem: KCor.from_fits(fits.open(kcor_name), trans_thresh) for kcor_name in kcor_names}

    magsys = CompositeMagSys(name, {
        band: kcor.zpoffs[name]
        for kcor in kcors.values()
        for name, band in kcor.bands.items()
    })

    bandmap = multivaluedict(
        ((survey, bandname[-1]), kcors[kcor_name.stem].bands[bandname])
        for kcor_name in kcor_names for header in [fits.open(kcor_name)[0].header]
        for i in range(header['NFILTERS']) for hkey in [f'FILT{i+1:0>3}'] for bandname in [header[hkey]]
        for survey in [header.comments[hkey].rsplit('SURVEY=', 1)[1]] if survey
    )

    return bandmap, magsys
