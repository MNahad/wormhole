# Copyright 2024-2025 Mohammed Nawabuddin
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field, InitVar
from enum import Enum
from functools import partial
import math
from os import path
from typing import Optional, SupportsIndex

import grain.python as grain
from jax import Array, numpy as jnp, random
import pyarrow as pa

import wormhole.common.defaults as defaults
import wormhole.common.npz as npz
import wormhole.common.pa_files as pa_files
from wormhole.common.utils import get_vertex_pairs_from_edges
from wormhole.config import config


@dataclass
class Data:
    data: InitVar[npz.Npz]
    time: Array = field(init=False)
    pdcsap_flux: Array = field(init=False)
    quality: Array = field(init=False)
    sequence_length: Array = field(init=False)

    def __post_init__(self, data: npz.Npz) -> None:
        self.time = data["TIME"]
        self.pdcsap_flux = data["PDCSAP_FLUX"]
        self.quality = data["QUALITY"]
        self.sequence_length = jnp.argmin(self.time)


class Label(Enum):
    TCE = True
    NO_TCE = False


@dataclass(frozen=True)
class Meta:
    sector: str
    ticid: str


@dataclass(frozen=True)
class LightCurve:
    data: Data
    label: Label
    meta: Meta


class _LightCurveDataSource:
    lc_dir: str
    _slice: tuple[int, int]
    _active_manifest: pa.Table
    _lc_path: str = path.join(*config()["data"]["catalogue"]["lc"]["path"])

    def __init__(
        self,
        lc_dir: str,
        manifest: pa.Table,
        shard: Optional[tuple[float, float]] = None,
    ) -> None:
        self.lc_dir = lc_dir
        manifest_size = manifest.num_rows
        self._slice = (
            (
                offset := math.ceil(manifest_size * shard[0]),
                math.ceil(manifest_size * shard[1]) - offset,
            )
            if shard
            else (0, manifest_size)
        )
        self._active_manifest = manifest.slice(*self._slice)

    def __len__(self) -> int:
        return self._active_manifest.num_rows

    def __getitem__(self, record_key: SupportsIndex) -> LightCurve:
        sector = str(self._active_manifest["sector"][record_key].as_py())
        ticid = str(self._active_manifest["ticid"][record_key].as_py())
        label = (
            Label.TCE
            if self._active_manifest["has_tce"][record_key].as_py()
            else Label.NO_TCE
        )
        return LightCurve(
            Data(
                npz.read(
                    path.join(
                        self.lc_dir, self._lc_path, sector, ticid + ".npz"
                    )
                )
            ),
            label,
            Meta(sector, ticid),
        )


class _LightCurveDataSourceFactory:
    manifest_dir: str
    lc_dir: str
    _manifest = pa.Table
    _manifest_schema: dict[str, pa.DataType] = defaults.metadataset_schema()
    _manifest_path: str = path.join(
        *(
            config()["data"]["catalogue"]["lc_manifest"]["path"]
            + ("manifest.csv",)
        )
    )
    _choice: partial[Array] = partial(random.choice, replace=False)

    def __init__(
        self,
        manifest_dir: str,
        lc_dir: str,
    ) -> None:
        self.lc_dir = lc_dir
        self.manifest_dir = manifest_dir
        manifest = pa_files.read_csv(
            path.join(self.manifest_dir, self._manifest_path),
            self._manifest_schema,
        )
        shuffled_indices = self._choice(
            random.key(0),
            jnp.arange(0, manifest.num_rows),
            shape=(manifest.num_rows,),
        )
        self._manifest = manifest.take(shuffled_indices.__array__())

    def new(
        self,
        shard: Optional[tuple[float, float]] = None,
    ) -> grain.RandomAccessDataSource:
        return _LightCurveDataSource(self.lc_dir, self._manifest, shard)


class LightCurveLoader:
    manifest_dir: str
    lc_dir: str
    _shards: tuple[tuple[float, float], ...]
    _batch_size: int
    _factory: _LightCurveDataSourceFactory
    _data_sources: tuple[grain.RandomAccessDataSource, ...]
    _data_loaders: tuple[grain.DataLoader, ...]

    def __init__(
        self,
        manifest_dir: str,
        lc_dir: str,
        split_ratio: Optional[tuple[float, ...]] = None,
        batch_size: Optional[int] = None,
    ) -> None:
        self.manifest_dir = manifest_dir
        self.lc_dir = lc_dir
        self._shards = (
            get_vertex_pairs_from_edges(split_ratio)
            if split_ratio
            else ((0.0, 1.0),)
        )
        self._batch_size = batch_size if batch_size else 0
        self._factory = _LightCurveDataSourceFactory(
            self.manifest_dir,
            self.lc_dir,
        )
        self._data_sources = tuple(
            self._factory.new(shard) for shard in self._shards
        )
        self._data_loaders = tuple(
            grain.DataLoader(
                data_source=data_source,
                sampler=grain.IndexSampler(
                    num_records=len(data_source),
                    shard_options=grain.NoSharding(),
                    shuffle=True,
                    num_epochs=1,
                    seed=0,
                ),
                operations=(
                    (grain.Batch(batch_size=self._batch_size),)
                    if self._batch_size > 0
                    else ()
                ),
            )
            for data_source in self._data_sources
        )

    def get(self) -> tuple[grain.DataLoader, ...]:
        return self._data_loaders
