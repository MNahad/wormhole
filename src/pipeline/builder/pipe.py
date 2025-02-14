# Copyright 2024 Mohammed Nawabuddin
# SPDX-License-Identifier: Apache-2.0

from os import path
from typing import Callable

from common.fs import iter_files, make_dir
from common.utils import chain


def ETLPipe[
    T
](
    extractor: Callable[[str, str], T],
    transformers: tuple[Callable[[T], T], ...],
    loader: Callable[[T, str, str], None],
    source: str,
    sink: str,
) -> Callable[[], None]:

    def run() -> None:

        def extractor_fn(file: str) -> Callable[[None], T]:
            return lambda _: extractor(file, source)

        def loader_fn(file: str) -> Callable[[T], None]:
            return lambda data: loader(data, file, sink)

        def piped_fn(file: str) -> Callable[[None], None]:
            return chain(
                extractor_fn(file),
                *transformers,
                f_final=loader_fn(file),
            )

        make_dir(sink)
        for file in iter_files(source):
            piped_fn(path.basename(file))(None)

    return run
