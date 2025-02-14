# Copyright 2024 Mohammed Nawabuddin
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from functools import partial
from os import path

from jax import Array, numpy as jnp, random
import pyarrow as pa
from pyarrow import compute as pc

import common.pa_files as pa_files


@dataclass
class MetaDataset:
    dataset_dir: str
    dataset: pa.Table = field(init=False)
    _dataset_schema: dict[str, pa.DataType] = field(
        default_factory=lambda: {
            "sector": pa.uint8(),
            "ticid": pa.uint64(),
            "url": pa.string(),
            "has_tce": pa.bool_(),
        },
        init=False,
    )
    _cache_path: str = field(default="lc/dataset/dataset.csv", init=False)
    _indices: tuple[int, int] = field(
        default_factory=lambda: (0, 0),
        init=False,
    )
    _partitioned_tce_indices: pa.Array = field(init=False)
    _partition_index: pa.Scalar = field(init=False)
    _takeable: Array = field(init=False)
    _key: Array = field(default_factory=lambda: random.key(0), init=False)

    @staticmethod
    def _choice(key: Array, array: Array, shape: int) -> Array:
        choice = partial(random.choice, replace=False)
        return choice(key, array, shape=(shape,))

    def __post_init__(self) -> None:
        self._load()
        self._partitioned_tce_indices = pc.array_sort_indices(
            self.dataset["has_tce"],
            order="descending",
        )
        first_non_tce_index = self.dataset["has_tce"].index(
            pa.scalar(False, type=pa.bool_()),
        )
        self._partition_index = self._partitioned_tce_indices.index(
            first_non_tce_index,
        )
        self._takeable = jnp.full(
            len(self._partitioned_tce_indices),
            True,
            dtype=bool,
        )

    def take(
        self,
        k: tuple[int, int],
        shuffle: bool = False,
    ) -> tuple[pa.Table, pa.Table]:
        partition_index = self._partition_index.as_py()
        if shuffle:
            takeable_tce = self._takeable[:partition_index]
            takeable_non_tce = self._takeable[partition_index:]
            self._key, key = random.split(self._key)
            takeable_tce_indices = MetaDataset._choice(
                key,
                takeable_tce.nonzero()[0],
                shape=k[0],
            )
            takeable_non_tce_indices = (
                MetaDataset._choice(
                    key,
                    takeable_non_tce.nonzero()[0],
                    shape=k[1],
                )
                + partition_index
            )
            self._takeable = self._takeable.at[
                jnp.concatenate(
                    [takeable_tce_indices, takeable_non_tce_indices],
                )
            ].set(False)
            tce_indices = self._partitioned_tce_indices.take(
                takeable_tce_indices.__array__(),
            )
            non_tce_indices = self._partitioned_tce_indices.take(
                takeable_non_tce_indices.__array__(),
            )
            return (
                self.dataset.take(tce_indices),
                self.dataset.take(non_tce_indices),
            )
        tce_partition_indices, non_tce_partition_indices = (
            list(range(self._indices[0], self._indices[0] + k[0])),
            list(
                range(
                    self._indices[1] + partition_index,
                    self._indices[1] + partition_index + k[1],
                )
            ),
        )
        self._indices = self._indices[0] + k[0], self._indices[1] + k[1]
        tce_indices = self._partitioned_tce_indices.take(tce_partition_indices)
        non_tce_indices = self._partitioned_tce_indices.take(
            non_tce_partition_indices
        )
        return (
            self.dataset.take(tce_indices),
            self.dataset.take(non_tce_indices),
        )

    def _load(self) -> None:
        self.dataset = pa_files.read_csv(
            path.join(self.dataset_dir, self._cache_path),
            self._dataset_schema,
        )
