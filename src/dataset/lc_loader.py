# Copyright 2024 Mohammed Nawabuddin
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from os import path
from typing import Iterator

from jax import Array
import pyarrow as pa

import common.fits as fits
from common.fs import make_dir

type Sector = pa.UInt8Scalar
type Ticid = pa.UInt64Scalar
type Url = pa.StringScalar


@dataclass
class LightCurveDataset:
    lc_dir: str
    meta: pa.Table
    _cache_path: str = field(default="lc/curves/", init=False)

    def __post_init__(self) -> None:
        make_dir(path.join(self.lc_dir, self._cache_path))

    def load(self) -> Iterator[tuple[Sector, Ticid, dict[str, Array]]]:
        return (
            (
                sector,
                ticid := self.meta["ticid"][i],
                self._load(sector, ticid, self.meta["url"][i]),
            )
            for i, sector in enumerate(self.meta["sector"])
        )

    def _load(
        self,
        sector: Sector,
        ticid: Ticid,
        url: Url,
    ) -> dict[str, Array]:
        keys = ["TIME", "PDCSAP_FLUX", "QUALITY"]
        if self._is_cached(sector, ticid):
            return fits.read(
                path.abspath(
                    path.join(
                        self.lc_dir,
                        self._cache_path,
                        str(sector.as_py()),
                        str(ticid.as_py()) + ".npz",
                    )
                ),
                keys,
            )
        lc = {k: v for k, v in fits.download(url.as_py(), 1, keys)}
        self._cache(sector, ticid, lc)
        return lc

    def _is_cached(self, sector: Sector, ticid: Ticid) -> bool:
        return path.exists(
            path.abspath(
                path.join(
                    self.lc_dir,
                    self._cache_path,
                    str(sector.as_py()),
                    str(ticid.as_py()) + ".npz",
                )
            )
        )

    def _cache(
        self,
        sector: Sector,
        ticid: Ticid,
        curve: dict[str, Array],
    ) -> None:
        make_dir(
            dir := path.join(
                self.lc_dir,
                self._cache_path,
                str(sector.as_py()),
            )
        )
        fits.write(path.join(dir, str(ticid.as_py()) + ".npz"), curve)
