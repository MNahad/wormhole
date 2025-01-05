# Copyright 2024 Mohammed Nawabuddin
# SPDX-License-Identifier: Apache-2.0

from typing import Callable

from common.fs import iter_files, make_dir
from common.utils import chain


class ETLPipe[T]:
    source: str
    sink: str
    extractor: Callable[[str, str], T]
    transformers: tuple[Callable[[T], T], ...]
    loader: Callable[[T, str], None]

    def __init__(
        self,
        extractor: Callable[[str, str], T],
        transformers: tuple[Callable[[T], T], ...],
        loader: Callable[[T, str], None],
        source: str,
        sink: str,
    ) -> None:
        self.source = source
        self.sink = sink
        self.extractor = extractor
        self.transformers = transformers
        self.loader = loader

    def run(self) -> None:

        def extractor() -> Callable[[str], T]:
            return lambda file: self.extractor(file, self.source)

        def loader() -> Callable[[T], None]:
            return lambda data: self.loader(data, self.sink)

        def piped() -> Callable[[str], None]:
            return chain(
                extractor(),
                *self.transformers,
                f_final=loader(),
            )

        make_dir(self.sink)
        piped_f = piped()
        for file in iter_files(self.source):
            piped_f(file)
