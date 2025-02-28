# Copyright 2024-2025 Mohammed Nawabuddin
# SPDX-License-Identifier: Apache-2.0

from os import path
from typing import Callable

from wormhole.config import config
from .bulk_fetch_stsci import fetch_bulk_data
from .generate_dataset import join_datasets
from .generate_manifest import generate_lightcurve_manifest
from .lightcurve_pipe import process_lightcurves
from .metadataset_pipe import process_metadataset
from .recache_lightcurves import recache_lightcurve_store


def get_ordered_names() -> list[str]:
    return list(
        stage[1].__name__
        for stage in sorted(_get_allowed_stages(), key=lambda x: x[0])
    )


def run(name: str) -> None:
    bronze_dir = path.join(
        *(
            config()["data"]["catalogue"]["path"]
            + config()["data"]["catalogue"]["bronze"]["path"]
        )
    )
    silver_dir = path.join(
        *(
            config()["data"]["catalogue"]["path"]
            + config()["data"]["catalogue"]["silver"]["path"]
        )
    )
    gold_dir = path.join(
        *(
            config()["data"]["catalogue"]["path"]
            + config()["data"]["catalogue"]["gold"]["path"]
        )
    )
    kwargs = {
        "bronze_dir": bronze_dir,
        "silver_dir": silver_dir,
        "gold_dir": gold_dir,
    }
    stages = _get_allowed_stages()
    next(
        filter(
            lambda stage: stage.__name__ == name,
            map(lambda stage: stage[1], stages),
        )
    )(**kwargs)


def _get_allowed_stages() -> tuple[tuple[int, Callable], ...]:
    return (
        (1, fetch_bulk_data),
        (2, process_metadataset),
        (3, join_datasets),
        (4, generate_lightcurve_manifest),
        (5, recache_lightcurve_store),
        (6, process_lightcurves),
    )
