# Copyright 2024 Mohammed Nawabuddin
# SPDX-License-Identifier: Apache-2.0

from os import path

from .builder.pipe import ETLPipe
from .stages import metadataset_stages as stages


def process_metadataset(bronze_dir: str, silver_dir: str) -> None:
    _generate_lc_csv(bronze_dir, silver_dir)
    _generate_tce_csv(bronze_dir, silver_dir)


def _generate_lc_csv(bronze_dir: str, silver_dir: str) -> None:
    pipe = ETLPipe(
        stages.lc_extractor,
        (stages.strip_leading_zeros,),
        stages.lc_loader,
        path.abspath(path.join(bronze_dir, "lc", "scripts")),
        path.abspath(path.join(silver_dir, "lc", "meta")),
    )
    pipe()


def _generate_tce_csv(bronze_dir: str, silver_dir: str) -> None:
    pipe = ETLPipe(
        stages.tce_extractor,
        (stages.get_unique_tces, stages.get_numeric_chars),
        stages.tce_loader,
        path.abspath(path.join(bronze_dir, "tce", "csvs")),
        path.abspath(path.join(silver_dir, "tce", "events")),
    )
    pipe()
