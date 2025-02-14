# Copyright 2024 Mohammed Nawabuddin
# SPDX-License-Identifier: Apache-2.0

from os import path

import pyarrow as pa

from common.fs import make_dir
import common.pa_files as pa_files
from dataset import MetaDataset


def generate_lightcurve_manifest(
    gold_dir: str,
    /,
    tce_ratio: tuple[int, int] = (20_000, 20_000),
) -> None:
    meta_dataset = MetaDataset(gold_dir)
    positives, negatives = meta_dataset.take(tce_ratio, shuffle=True)
    manifest_path = "lc/manifest/manifest.csv"
    make_dir(dir := path.join(gold_dir, path.dirname(manifest_path)))
    pa_files.write_csv(
        path.join(dir, path.basename(manifest_path)),
        pa.concat_tables([positives, negatives]),
    )
