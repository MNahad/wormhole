# Copyright 2024-2025 Mohammed Nawabuddin
# SPDX-License-Identifier: Apache-2.0

from argparse import ArgumentParser, Namespace

from wormhole.training import run


def add_training_parser(subparser: ArgumentParser) -> None:
    subparser.set_defaults(fn=handler)
    subparser.add_argument("train", help="start or resume a training job")
    subparser.add_argument("--id", help="the job id")


def handler(args: Namespace) -> None:
    run(args.id)
