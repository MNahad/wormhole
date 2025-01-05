# Copyright 2024 Mohammed Nawabuddin
# SPDX-License-Identifier: Apache-2.0

from os import listdir, makedirs, path
from typing import Iterator


def make_dir(dir: str) -> None:
    if not path.exists(path.abspath(dir)):
        makedirs(dir)


def iter_files(dir: str) -> Iterator[str]:
    for file in [
        path.abspath(path.join(dir, file))
        for file in sorted(listdir(path.abspath(dir)))
    ]:
        if not path.isfile(file):
            continue
        yield file


def swap_dir_of_file(
    file: str,
    new_dir: str,
    new_ext: str | None = None,
) -> str:
    new_file = path.abspath(path.join(new_dir, path.basename(file)))
    if new_ext:
        return path.splitext(new_file)[0] + new_ext
    return new_file


def yield_file(file: str) -> Iterator[str]:
    with open(file, "r") as f:
        yield from f
