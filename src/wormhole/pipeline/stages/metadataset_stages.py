# Copyright 2024 Mohammed Nawabuddin
# SPDX-License-Identifier: Apache-2.0

from os import path
import re
from typing import Iterable

import wormhole.common.csv_files as csv
from wormhole.common.fs import swap_dir_of_file, yield_file
from wormhole.common.utils import filter_no_prefix, match_substring, unique

type Pipe = Iterable[dict[str, str]]


def lc_extractor(file: str, source: str) -> Pipe:
    iter_lines = yield_file(path.join(source, file))
    urls = match_substring(
        iter_lines,
        re.compile(
            r"https://.+/tess[0-9]+-s([0-9]+)-([0-9]+).+\.fits$",
        ),
    )
    return (
        {
            "sector": url.group(1),
            "ticid": url.group(2),
            "url": url.group(0),
        }
        for url in urls
    )


def strip_leading_zeros(data: Pipe) -> Pipe:
    return (
        {
            "sector": contents["sector"].lstrip("0"),
            "ticid": contents["ticid"].lstrip("0"),
            "url": contents["url"],
        }
        for contents in data
    )


def lc_loader(data: Pipe, file: str, sink: str) -> None:
    csv.write(
        swap_dir_of_file(file, sink, new_ext=".csv"),
        ["sector", "ticid", "url"],
        data,
    )


def tce_extractor(file: str, source: str) -> Pipe:
    return csv.read(
        path.join(source, file),
        lambda arg: filter_no_prefix(arg, "#"),
    )


def get_unique_tces(data: Pipe) -> Pipe:
    return unique(data, "ticid")


def get_numeric_chars(data: Pipe) -> Pipe:
    return (
        {
            "sector": contents["sectors"][1:].lstrip("0"),
            "ticid": contents["ticid"],
        }
        for contents in data
    )


def tce_loader(data: Pipe, file: str, sink: str) -> None:
    csv.write(
        swap_dir_of_file(file, sink, new_ext=".csv"),
        ["sector", "ticid"],
        data,
    )
