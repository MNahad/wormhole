# Copyright 2024-2025 Mohammed Nawabuddin
# SPDX-License-Identifier: Apache-2.0

from functools import partial
import math
from os import path
from typing import Iterator, Optional, SupportsIndex

import grain.python as grain
from jax import Array, numpy as jnp, random
import pyarrow as pa

import wormhole.common.defaults as defaults
import wormhole.common.npz as npz
import wormhole.common.pa_files as pa_files
from wormhole.common.utils import get_vertex_pairs_from_edges
from wormhole.config import config
from . import model, transforms


class _DataSource:
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

    def __repr__(self) -> str:
        return f"_DataSource(_slice={self._slice})"

    def __len__(self) -> int:
        return self._active_manifest.num_rows

    def __getitem__(self, record_key: SupportsIndex) -> model.LightCurve:
        sector = str(self._active_manifest["sector"][record_key].as_py())
        ticid = str(self._active_manifest["ticid"][record_key].as_py())
        return model.LightCurve(
            model.Data(
                npz.read(
                    path.join(
                        self.lc_dir, self._lc_path, sector, ticid + ".npz"
                    )
                )
            ),
            model.Meta(
                sector,
                ticid,
                self._active_manifest["has_tce"][record_key].as_py(),
            ),
        )


class _DataSourceFactory:
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
        return _DataSource(self.lc_dir, self._manifest, shard)


class DataLoader:
    manifest_dir: str
    lc_dir: str
    _shards: tuple[tuple[float, float], ...]
    _batch_size: int
    _num_epochs: int
    _factory: _DataSourceFactory
    _data_sources: tuple[grain.RandomAccessDataSource, ...]
    _data_loaders: tuple[grain.DataLoader, ...]
    _allowed_labels_shard_indices: dict[int, tuple[bool, ...]]

    def __init__(
        self,
        manifest_dir: str,
        lc_dir: str,
        *,
        split_ratio: Optional[tuple[float, ...]] = None,
        allowed_labels_split_indices: Optional[
            dict[int, tuple[bool, ...]]
        ] = None,
        batch_size: Optional[int] = None,
        num_epochs: Optional[int] = None,
    ) -> None:
        self.manifest_dir = manifest_dir
        self.lc_dir = lc_dir
        self._shards = (
            get_vertex_pairs_from_edges(split_ratio)
            if split_ratio
            else ((0.0, 1.0),)
        )
        self._allowed_labels_shard_indices = (
            allowed_labels_split_indices
            if allowed_labels_split_indices
            else {i: (True, False) for i in range(len(self._shards))}
        )
        self._batch_size = batch_size if batch_size else 0
        self._num_epochs = num_epochs if num_epochs else 1
        self._factory = _DataSourceFactory(
            self.manifest_dir,
            self.lc_dir,
        )
        self._data_sources = tuple(
            self._factory.new(shard) for shard in self._shards
        )
        operations: grain.Transformations = (
            transforms.PytreeTransform(),
            transforms.StackTransform(),
        )
        if self._batch_size > 0:
            operations += (grain.Batch(batch_size=self._batch_size),)
        self._data_loaders = tuple(
            grain.DataLoader(
                data_source=data_source,
                sampler=grain.IndexSampler(
                    num_records=len(data_source),
                    shard_options=grain.NoSharding(),
                    shuffle=True,
                    num_epochs=self._num_epochs,
                    seed=0,
                ),
                operations=(
                    (
                        transforms.FilterLabels(
                            self._allowed_labels_shard_indices[i]
                        ),
                    )
                    + operations
                    if i in self._allowed_labels_shard_indices
                    and not (
                        True in self._allowed_labels_shard_indices[i]
                        and False in self._allowed_labels_shard_indices[i]
                    )
                    else operations
                ),
            )
            for i, data_source in enumerate(self._data_sources)
        )

    def get(self) -> tuple[Iterator[transforms.Pytree], ...]:
        return self._data_loaders
