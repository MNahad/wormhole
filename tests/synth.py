# Copyright 2024-2025 Mohammed Nawabuddin
# SPDX-License-Identifier: Apache-2.0

from os import path
import shutil

from wormhole.config import config


def use_tmp_dir() -> None:
    conf = config()
    bronze_dir = conf["data"]["catalogue"]["bronze"]["path"]
    silver_dir = conf["data"]["catalogue"]["silver"]["path"]
    gold_dir = conf["data"]["catalogue"]["gold"]["path"]
    shutil.copytree(
        "tests/data_input/bulk_lc",
        path.join(
            "tmp/data",
            *bronze_dir,
            *conf["data"]["catalogue"]["bulk_lc"]["path"],
        ),
        dirs_exist_ok=True,
    )
    shutil.copytree(
        "tests/data_input/bulk_tce",
        path.join(
            "tmp/data",
            *bronze_dir,
            *conf["data"]["catalogue"]["bulk_tce"]["path"],
        ),
        dirs_exist_ok=True,
    )
    shutil.copytree(
        "tests/data_input/lc",
        path.join(
            "tmp/data",
            *silver_dir,
            *conf["data"]["catalogue"]["lc"]["path"],
        ),
        dirs_exist_ok=True,
    )
    shutil.copytree(
        "tests/data_baseline/lc_metadata",
        path.join(
            "tmp/data_baseline",
            *silver_dir,
            *conf["data"]["catalogue"]["lc_metadata"]["path"],
        ),
        dirs_exist_ok=True,
    )
    shutil.copytree(
        "tests/data_baseline/tce_metadata",
        path.join(
            "tmp/data_baseline",
            *silver_dir,
            *conf["data"]["catalogue"]["tce_metadata"]["path"],
        ),
        dirs_exist_ok=True,
    )
    shutil.copytree(
        "tests/data_baseline/metadataset",
        path.join(
            "tmp/data_baseline",
            *gold_dir,
            *conf["data"]["catalogue"]["metadataset"]["path"],
        ),
        dirs_exist_ok=True,
    )
    shutil.copytree(
        "tests/data_baseline/lc_manifest",
        path.join(
            "tmp/data_baseline",
            *gold_dir,
            *conf["data"]["catalogue"]["lc_manifest"]["path"],
        ),
        dirs_exist_ok=True,
    )
    shutil.copytree(
        "tests/data_baseline/lc",
        path.join(
            "tmp/data_baseline",
            *gold_dir,
            *conf["data"]["catalogue"]["lc"]["path"],
        ),
        dirs_exist_ok=True,
    )


def clear_tmp_dir() -> None:
    shutil.rmtree("tmp/")


def get_tmp_dir() -> str:
    return "tmp/data/"


def get_baseline_dir() -> str:
    return "tmp/data_baseline/"
