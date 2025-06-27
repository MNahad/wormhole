# Copyright 2024-2025 Mohammed Nawabuddin
# SPDX-License-Identifier: Apache-2.0

from argparse import ArgumentParser, Namespace

from wormhole.training import run


def add_training_parser(subparser: ArgumentParser) -> None:
    subparser.set_defaults(fn=handler)
    training_subparsers = subparser.add_subparsers(help="training commands")
    train_parser = training_subparsers.add_parser(
        "train",
        help="start or resume a training job",
    )
    train_parser.add_argument("--id", help="the job id")
    eval_parser = training_subparsers.add_parser(
        "evaluate",
        help="evaluate the model of a training job",
    )
    eval_parser.add_argument("--eval-id", help="the existing job id")


def handler(args: Namespace) -> None:
    if "eval_id" in args:
        run(args.eval_id, mode="evaluate")
    else:
        run(args.id)
