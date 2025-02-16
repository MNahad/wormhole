# Copyright 2024 Mohammed Nawabuddin
# SPDX-License-Identifier: Apache-2.0

from os import path

from wormhole.common.fs import iter_files
from wormhole.config import config
from .builder.pipe import ETLPipe
from .stages import lightcurve_stages as stages


def process_lightcurves(silver_dir: str, gold_dir: str) -> None:
    curves_in_dir = path.abspath(
        path.join(silver_dir, *config()["data"]["catalogue"]["lc"]["path"])
    )
    curves_out_dir = path.abspath(
        path.join(gold_dir, *config()["data"]["catalogue"]["lc"]["path"])
    )
    for sector_dir in iter_files(curves_in_dir, lambda dir: path.isdir(dir)):
        _run_pipe(
            sector_dir,
            path.join(curves_out_dir, path.basename(sector_dir)),
        )


def _run_pipe(read_dir: str, write_dir: str) -> None:
    pipe = ETLPipe(
        stages.extract,
        (
            stages.filter_low_q,
            stages.standardise_t,
            stages.standardise_flux,
            stages.filter_nan,
        ),
        stages.load,
        read_dir,
        write_dir,
    )
    pipe()
