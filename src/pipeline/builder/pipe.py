# Copyright 2024 Mohammed Nawabuddin
# SPDX-License-Identifier: Apache-2.0

from os import path
from typing import Callable

from common.fs import iter_files, make_dir
from common.utils import chain


class ETLPipe[T]:
    source: str
    sink: str
    extractor: Callable[[str, str], T]
    transformers: tuple[Callable[[T], T], ...]
    loader: Callable[[T, str, str], None]

    def __init__(
        self,
        extractor: Callable[[str, str], T],
        transformers: tuple[Callable[[T], T], ...],
        loader: Callable[[T, str, str], None],
        source: str,
        sink: str,
    ) -> None:
        self.source = source
        self.sink = sink
        self.extractor = extractor
        self.transformers = transformers
        self.loader = loader

    def run(self) -> None:

        def extractor(file: str) -> Callable[[None], T]:
            return lambda _: self.extractor(file, self.source)

        def loader(file: str) -> Callable[[T], None]:
            return lambda data: self.loader(data, file, self.sink)

        def piped(file: str) -> Callable[[None], None]:
            return chain(
                extractor(file),
                *self.transformers,
                f_final=loader(file),
            )

        make_dir(self.sink)
        for file in iter_files(self.source):
            piped(path.basename(file))(None)
