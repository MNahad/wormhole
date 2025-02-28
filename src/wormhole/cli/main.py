# Copyright 2024-2025 Mohammed Nawabuddin
# SPDX-License-Identifier: Apache-2.0

import argparse

from .pipeline import add_pipeline_parser


def entrypoint():
    parser = argparse.ArgumentParser(
        prog="wormhole",
        description="Exoplanet detection with Flaxoil",
    )
    subparsers = parser.add_subparsers()
    pipeline_parser = subparsers.add_parser(name="pipeline")
    add_pipeline_parser(pipeline_parser)
    args = parser.parse_args()
    if "fn" not in args:
        print(parser.format_help())
        return
    args.fn(args)
