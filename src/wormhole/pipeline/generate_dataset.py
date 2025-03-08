# Copyright 2024-2025 Mohammed Nawabuddin
# SPDX-License-Identifier: Apache-2.0

from os import path

import pyarrow as pa

import wormhole.common.defaults as defaults
from wormhole.common.fs import make_dir
import wormhole.common.pa_files as pa_files
from wormhole.config import config


def join_datasets(silver_dir: str, gold_dir: str, **kwargs) -> None:
    lc_path = path.join(
        silver_dir,
        *config()["data"]["catalogue"]["lc_metadata"]["path"],
    )
    tce_path = path.join(
        silver_dir,
        *config()["data"]["catalogue"]["tce_metadata"]["path"],
    )
    dataset_path = path.join(
        *(
            config()["data"]["catalogue"]["metadataset"]["path"]
            + ("metadataset.csv",)
        )
    )
    schema = defaults.metadataset_schema()
    lc_table = pa_files.read_csv(
        lc_path,
        {k: schema[k] for k in ["sector", "ticid", "url"]},
    )
    tce_table = pa_files.read_csv(
        tce_path,
        {k: schema[k] for k in ["sector", "ticid"]},
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
    joined_table = joined_table.sort_by(
        [("sector", "ascending"), ("ticid", "ascending")]
    )
    make_dir(dir := path.join(gold_dir, path.dirname(dataset_path)))
    pa_files.write_csv(
        path.join(dir, path.basename(dataset_path)),
        joined_table,
    )
