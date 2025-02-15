# Copyright 2024 Mohammed Nawabuddin
# SPDX-License-Identifier: Apache-2.0

from os import path

import pyarrow as pa

from wormhole.common.fs import make_dir
import wormhole.common.pa_files as pa_files


def join_datasets(silver_dir: str, gold_dir: str) -> None:
    lc_path = path.join(silver_dir, "lc", "meta")
    tce_path = path.join(silver_dir, "tce", "events")
    dataset_path = "lc/dataset/dataset.csv"
    schema = {
        "sector": pa.uint8(),
        "ticid": pa.uint64(),
        "url": pa.string(),
        "has_tce": pa.bool_(),
    }
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
