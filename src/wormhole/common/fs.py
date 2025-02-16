# Copyright 2024-2025 Mohammed Nawabuddin
# SPDX-License-Identifier: Apache-2.0

from os import listdir, makedirs, path
from typing import Callable, Iterator, Optional


def make_dir(dir: str) -> None:
    if not path.exists(path.abspath(dir)):
        makedirs(dir)


def iter_files(
    dir: str,
    predicate: Optional[Callable[[str], bool]] = None,
) -> Iterator[str]:
    predicate = predicate if predicate else lambda file: path.isfile(file)
    return (
        path.abspath(path.join(dir, file))
        for file in sorted(listdir(path.abspath(dir)))
        if predicate(path.abspath(path.join(dir, file)))
    )


def swap_dir_of_file(
    file: str,
    new_dir: str,
    new_ext: Optional[str] = None,
) -> str:
    new_file = path.abspath(path.join(new_dir, path.basename(file)))
    if new_ext:
        return path.splitext(new_file)[0] + new_ext
    return new_file


def yield_file(file: str) -> Iterator[str]:
    with open(file, "r") as f:
        yield from f
