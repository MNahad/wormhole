# Copyright 2024 Mohammed Nawabuddin
# SPDX-License-Identifier: Apache-2.0

from os import path

import pyarrow as pa
from pyarrow import csv

from wormhole.common.fs import iter_files


def read_csv(target_path: str, schema: dict[str, pa.DataType]) -> pa.Table:
    if path.isfile(csv_path := path.abspath(target_path)):
        return _read_csv(csv_path, schema)
    return pa.concat_tables(
        (_read_csv(file, schema) for file in iter_files(csv_path)),
    )


def write_csv(target_path: str, data: pa.Table) -> None:
    csv.write_csv(data, path.abspath(target_path))


def _read_csv(target_file: str, schema: dict[str, pa.DataType]) -> pa.Table:
    return csv.read_csv(
        target_file,
        convert_options=csv.ConvertOptions(column_types=pa.schema(schema)),
    )
