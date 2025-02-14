# Copyright 2024 Mohammed Nawabuddin
# SPDX-License-Identifier: Apache-2.0

from collections import deque
from dataclasses import dataclass, field
from itertools import batched
import math
from os import path

from jax import Array
import pyarrow as pa

import wormhole.common.async_wrapper as async_wrapper
import wormhole.common.fits as fits
from wormhole.common.fs import make_dir
import wormhole.common.pa_files as pa_files

type Sector = pa.UInt8Scalar
type Ticid = pa.UInt64Scalar
type Url = pa.StringScalar


@dataclass
class LightCurveStore:
    manifest_dir: str
    lc_dir: str
    _meta: pa.Table = field(init=False)
    _meta_schema: dict[str, pa.DataType] = field(
        default_factory=lambda: {
            "sector": pa.uint8(),
            "ticid": pa.uint64(),
            "url": pa.string(),
            "has_tce": pa.bool_(),
        },
        init=False,
    )
    _manifest_path: str = field(default="lc/manifest/manifest.csv", init=False)
    _cache_path: str = field(default="lc/curves/", init=False)
    _data_keys: list[str] = field(
        default_factory=lambda: [
            "TIME",
            "PDCSAP_FLUX",
            "QUALITY",
        ],
        init=False,
    )
    _async_batch_size: int = field(default=100, init=False)

    @staticmethod
    def _is_cached(
        sector: Sector,
        ticid: Ticid,
        lc_dir: str,
        cache_path: str,
    ) -> bool:
        return path.exists(
            path.abspath(
                path.join(
                    lc_dir,
                    cache_path,
                    str(sector.as_py()),
                    str(ticid.as_py()) + ".npz",
                )
            )
        )

    @staticmethod
    def _cache(
        sector: Sector,
        ticid: Ticid,
        curve: dict[str, Array],
        lc_dir: str,
        cache_path: str,
    ) -> None:
        fits.write(
            path.join(
                lc_dir,
                cache_path,
                str(sector.as_py()),
                str(ticid.as_py()) + ".npz",
            ),
            curve,
        )

    @staticmethod
    def _download(
        meta: tuple[Sector, Ticid, Url],
        data_keys: list[str],
        lc_dir: str,
        cache_path: str,
    ) -> None:
        sector, ticid, url = meta
        if not LightCurveStore._is_cached(sector, ticid, lc_dir, cache_path):
            lc = fits.download(url.as_py(), 1, data_keys)
            LightCurveStore._cache(sector, ticid, lc, lc_dir, cache_path)

    def __post_init__(self) -> None:
        self._meta = pa_files.read_csv(
            path.join(self.manifest_dir, self._manifest_path),
            self._meta_schema,
        )
        make_dir(cache_dir := path.join(self.lc_dir, self._cache_path))
        for sector in self._meta["sector"].unique():
            make_dir(path.join(cache_dir, str(sector.as_py())))

    def download(self, debug=False) -> None:
        batches = batched(
            (
                (
                    sector,
                    self._meta["ticid"][i],
                    self._meta["url"][i],
                )
                for i, sector in enumerate(self._meta["sector"])
            ),
            self._async_batch_size,
        )
        if debug:
            debug_total = math.ceil(
                self._meta.num_rows / self._async_batch_size
            )
            debug_count = 0
        for batch in batches:
            if debug:
                debug_count += 1
                print(f"DOWNLOADING LC BATCH {debug_count} OF {debug_total}")
            downloaded = async_wrapper.exec(
                LightCurveStore._download,
                batch,
                self._data_keys,
                self.lc_dir,
                self._cache_path,
            )
            deque(downloaded, maxlen=0)
