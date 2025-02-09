# Copyright 2024 Mohammed Nawabuddin
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from functools import partial
from os import path
from typing import Optional

from jax import Array, numpy as jnp, random
import pyarrow as pa
from pyarrow import compute as pc, csv

from common.fs import iter_files, make_dir


@dataclass
class MetaDataset:
    dataset_dir: str
    source_dir: Optional[str] = field(default=None)
    re_cache: bool = field(default=False)
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

    @staticmethod
    def _read_csv(file: str, schema: dict[str, pa.DataType]) -> pa.Table:
        return csv.read_csv(
            file,
            convert_options=csv.ConvertOptions(
                column_types=pa.schema(schema),
            ),
        )

    @staticmethod
    def _load_table_from_csvs(
        dir: str,
        schema: dict[str, pa.DataType],
    ) -> pa.Table:
        file_it = iter_files(dir)
        next_file = next(file_it, "")
        if len(next_file):
            table = MetaDataset._read_csv(next_file, schema)
        for file in file_it:
            table = pa.concat_tables(
                [
                    table,
                    MetaDataset._read_csv(file, schema),
                ]
            )
        return table

    def __post_init__(self) -> None:
        if (not self.re_cache) and path.exists(
            path.abspath(path.join(self.dataset_dir, self._cache_path))
        ):
            self._load()
        else:
            self.generate()
            self.save()
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

    def generate(self) -> None:
        source_dir = self.source_dir if self.source_dir else self.dataset_dir
        lc_path = path.join(source_dir, "lc", "meta")
        tce_path = path.join(source_dir, "tce", "events")
        lc_table = MetaDataset._load_table_from_csvs(
            lc_path,
            {k: self._dataset_schema[k] for k in ["sector", "ticid", "url"]},
        )
        tce_table = MetaDataset._load_table_from_csvs(
            tce_path,
            {k: self._dataset_schema[k] for k in ["sector", "ticid"]},
        )
        tce_table = tce_table.append_column(
            "has_tce",
            pa.array(
                (True for _ in range(tce_table.num_rows)),
                type=pa.bool_(),
            ),
        )
        joined_table = lc_table.join(
            tce_table,
            keys=["sector", "ticid"],
            join_type="left outer",
        )
        new_tce_array = joined_table["has_tce"].fill_null(
            pa.scalar(False, type=pa.bool_())
        )
        joined_table = joined_table.set_column(3, "has_tce", new_tce_array)
        self.dataset = joined_table.sort_by(
            [("sector", "ascending"), ("ticid", "ascending")]
        )

    def save(self) -> None:
        make_dir(
            dir := path.join(self.dataset_dir, path.dirname(self._cache_path))
        )
        csv.write_csv(
            self.dataset,
            path.abspath(path.join(dir, path.basename(self._cache_path))),
        )

    def _load(self) -> None:
        self.dataset = self._read_csv(
            path.abspath(path.join(self.dataset_dir, self._cache_path)),
            schema=self._dataset_schema,
        )
