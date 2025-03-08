# Copyright 2024-2025 Mohammed Nawabuddin
# SPDX-License-Identifier: Apache-2.0

from os import path

import pyarrow as pa

from wormhole.common.fs import make_dir
import wormhole.common.pa_files as pa_files
from wormhole.config import config
from wormhole.dataset import MetaDataset


def generate_lightcurve_manifest(gold_dir: str, **kwargs) -> None:
    meta_dataset = MetaDataset(gold_dir)
    tce_sample_ratio = config()["pipelines"]["args"]["tce_sample_ratio"]
    positives, negatives = meta_dataset.take(tce_sample_ratio, shuffle=True)
    manifest_path = path.join(
        *(
            config()["data"]["catalogue"]["lc_manifest"]["path"]
            + ("manifest.csv",)
        )
    )
    make_dir(dir := path.join(gold_dir, path.dirname(manifest_path)))
    pa_files.write_csv(
        path.join(dir, path.basename(manifest_path)),
        pa.concat_tables([positives, negatives]),
    )
