# Copyright 2024-2025 Mohammed Nawabuddin
# SPDX-License-Identifier: Apache-2.0

from functools import partial
from os import path
import shutil

from wormhole.config import config


def use_tmp_dir() -> None:
    conf = config()
    bronze_dir = conf["data"]["catalogue"]["bronze"]["path"]
    silver_dir = conf["data"]["catalogue"]["silver"]["path"]
    gold_dir = conf["data"]["catalogue"]["gold"]["path"]
    copytree = partial(shutil.copytree, dirs_exist_ok=True)
    copytree(
        "tests/data_input/bulk_lc",
        path.join(
            "tmp/data",
            *bronze_dir,
            *conf["data"]["catalogue"]["bulk_lc"]["path"],
        ),
    )
    copytree(
        "tests/data_input/bulk_tce",
        path.join(
            "tmp/data",
            *bronze_dir,
            *conf["data"]["catalogue"]["bulk_tce"]["path"],
        ),
    )
    copytree(
        "tests/data_input/lc",
        path.join(
            "tmp/data",
            *silver_dir,
            *conf["data"]["catalogue"]["lc"]["path"],
        ),
    )
    copytree(
        "tests/data_baseline/lc_metadata",
        path.join(
            "tmp/data_baseline",
            *silver_dir,
            *conf["data"]["catalogue"]["lc_metadata"]["path"],
        ),
    )
    copytree(
        "tests/data_baseline/tce_metadata",
        path.join(
            "tmp/data_baseline",
            *silver_dir,
            *conf["data"]["catalogue"]["tce_metadata"]["path"],
        ),
    )
    copytree(
        "tests/data_baseline/metadataset",
        path.join(
            "tmp/data_baseline",
            *gold_dir,
            *conf["data"]["catalogue"]["metadataset"]["path"],
        ),
    )
    copytree(
        "tests/data_baseline/lc_manifest",
        path.join(
            "tmp/data_baseline",
            *gold_dir,
            *conf["data"]["catalogue"]["lc_manifest"]["path"],
        ),
    )
    copytree(
        "tests/data_baseline/lc",
        path.join(
            "tmp/data_baseline",
            *gold_dir,
            *conf["data"]["catalogue"]["lc"]["path"],
        ),
    )


def clear_tmp_dir() -> None:
    shutil.rmtree("tmp/")


def get_tmp_dir() -> str:
    return "tmp/data/"


def get_baseline_dir() -> str:
    return "tmp/data_baseline/"
