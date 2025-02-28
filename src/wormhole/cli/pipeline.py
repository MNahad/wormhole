# Copyright 2024-2025 Mohammed Nawabuddin
# SPDX-License-Identifier: Apache-2.0

from argparse import ArgumentParser, Namespace

from wormhole.pipeline import get_ordered_names, run


def add_pipeline_parser(subparser: ArgumentParser) -> None:
    subparser.set_defaults(fn=handler)
    subparser.add_argument("run", help="run a pipeline")
    me_group = subparser.add_mutually_exclusive_group()
    me_group.add_argument(
        "--only",
        choices=get_ordered_names(),
        metavar="NAME",
        help="only run a single stage by name",
    )
    me_group.add_argument(
        "--all",
        "-a",
        action="store_true",
        help="run all stages",
    )


def handler(args: Namespace) -> None:
    if args.all:
        for fn in get_ordered_names():
            print(f"RUNNING {fn} ...")
            run(fn)
    else:
        print(f"RUNNING {args.only} ...")
        run(args.only)
