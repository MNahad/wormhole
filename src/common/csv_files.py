# Copyright 2024 Mohammed Nawabuddin
# SPDX-License-Identifier: Apache-2.0

import csv
from typing import Callable, Iterable, Iterator


class CSV:

    @staticmethod
    def read(
        file: str,
        pre_transform: Callable[[Iterable[str]], Iterator[str]] | None = None,
    ) -> Iterator[dict[str, str]]:
        with open(file, "r") as in_f:
            input = pre_transform(in_f) if pre_transform else in_f
            reader = csv.DictReader(input)
            for row in reader:
                yield row

    @staticmethod
    def write(
        file: str,
        header: list[str],
        data: Iterable[dict[str, str]],
    ) -> None:
        with open(file, "w") as out_f:
            writer = csv.DictWriter(out_f, header, dialect=csv.unix_dialect())
            writer.writeheader()
            writer.writerows(data)
