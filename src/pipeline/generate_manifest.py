# Copyright 2024 Mohammed Nawabuddin
# SPDX-License-Identifier: Apache-2.0

from os import path

import pyarrow as pa
from pyarrow import csv

from common.fs import make_dir
from dataset import MetaDataset


def generate_lightcurve_manifest(
    bronze_dir: str,
    silver_dir: str,
    gold_dir: str,
    /,
    lc_ratio: tuple[int, int] = (20_000, 20_000),
) -> None:
    meta_dataset = MetaDataset(gold_dir, silver_dir)
    positives, negatives = meta_dataset.take(lc_ratio, shuffle=True)
    manifest_path = "lc/manifest/manifest.csv"
    make_dir(dir := path.join(bronze_dir, path.dirname(manifest_path)))
    csv.write_csv(
        pa.concat_tables([positives, negatives]),
        path.abspath(path.join(dir, path.basename(manifest_path))),
    )
