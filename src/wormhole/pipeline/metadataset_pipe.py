# Copyright 2024-2025 Mohammed Nawabuddin
# SPDX-License-Identifier: Apache-2.0

from os import path

from wormhole.config import config
from .builder.pipe import ETLPipe
from .stages import metadataset_stages as stages


def process_metadataset(bronze_dir: str, silver_dir: str, **kwargs) -> None:
    _generate_lc_csv(bronze_dir, silver_dir)
    _generate_tce_csv(bronze_dir, silver_dir)


def _generate_lc_csv(bronze_dir: str, silver_dir: str) -> None:
    pipe = ETLPipe(
        stages.lc_extractor,
        (stages.strip_leading_zeros,),
        stages.lc_loader,
        path.abspath(
            path.join(
                bronze_dir,
                *config()["data"]["catalogue"]["bulk_lc"]["path"],
            )
        ),
        path.abspath(
            path.join(
                silver_dir,
                *config()["data"]["catalogue"]["lc_metadata"]["path"],
            )
        ),
    )
    pipe()


def _generate_tce_csv(bronze_dir: str, silver_dir: str) -> None:
    pipe = ETLPipe(
        stages.tce_extractor,
        (stages.get_unique_tces, stages.get_numeric_chars),
        stages.tce_loader,
        path.abspath(
            path.join(
                bronze_dir,
                *config()["data"]["catalogue"]["bulk_tce"]["path"],
            )
        ),
        path.abspath(
            path.join(
                silver_dir,
                *config()["data"]["catalogue"]["tce_metadata"]["path"],
            )
        ),
    )
    pipe()
