# Copyright 2024 Mohammed Nawabuddin
# SPDX-License-Identifier: Apache-2.0

from os import path
import re
from typing import Iterable

from common.csv_files import CSV
from common.fs import swap_dir_of_file, yield_file
from common.utils import filter_no_prefix, match_substring, unique
from .processing.pipe import ETLPipe
from .stsci_client.fetcher import FITS, STScI

type PipeE = str
type PipeT = tuple[Iterable[dict[str, str]], str]
type PipeL = None


def download(bronze_dir: str):
    stsci_client = STScI(bronze_dir)
    stsci_client.download()


def download_fits(bronze_dir: str):
    fits_client = FITS(bronze_dir)
    fits_client.download()


def process(bronze_dir: str, silver_dir: str):
    _generate_lc_csv(bronze_dir, silver_dir)
    _generate_tce_csv(bronze_dir, silver_dir)


def _generate_lc_csv(bronze_dir: str, silver_dir: str) -> None:
    bronze_dir = path.abspath(path.join(bronze_dir, "lc", "scripts"))
    silver_dir = path.abspath(path.join(silver_dir, "lc", "meta"))

    def extractor(file: PipeE, _: str) -> PipeT:
        iter_lines = yield_file(file)
        urls = match_substring(
            iter_lines,
            re.compile(
                r"https://.+/tess[0-9]+-s([0-9]+)-([0-9]+).+\.fits$",
            ),
        )
        return (
            (
                {
                    "sector": url.group(1),
                    "ticid": url.group(2),
                    "url": url.group(0),
                }
                for url in urls
            ),
            file,
        )

    def lstrip(data: PipeT) -> PipeT:
        return (
            (
                {
                    "sector": contents["sector"].lstrip("0"),
                    "ticid": contents["ticid"].lstrip("0"),
                    "url": contents["url"],
                }
                for contents in data[0]
            ),
            data[1],
        )

    def loader(data: PipeT, sink: str) -> PipeL:
        CSV.write(
            swap_dir_of_file(data[1], sink, new_ext=".csv"),
            ["sector", "ticid", "url"],
            data[0],
        )

    pipe = ETLPipe(
        extractor,
        (lstrip,),
        loader,
        bronze_dir,
        silver_dir,
    )
    pipe.run()


def _generate_tce_csv(bronze_dir: str, silver_dir: str) -> None:
    bronze_dir = path.abspath(path.join(bronze_dir, "tce", "csvs"))
    silver_dir = path.abspath(path.join(silver_dir, "tce", "events"))

    def extractor(file: PipeE, _: str) -> PipeT:
        raw_data = CSV.read(
            file,
            lambda arg: filter_no_prefix(arg, "#"),
        )
        return raw_data, file

    def unique_tces(data: PipeT) -> PipeT:
        return (unique(data[0], "ticid"), data[1])

    def prep_values(data: PipeT) -> PipeT:
        return (
            (
                {
                    "sector": contents["sectors"][1:].lstrip("0"),
                    "ticid": contents["ticid"],
                }
                for contents in data[0]
            ),
            data[1],
        )

    def loader(data: PipeT, sink: str) -> PipeL:
        CSV.write(
            swap_dir_of_file(data[1], sink, new_ext=".csv"),
            ["sector", "ticid"],
            data[0],
        )

    pipe = ETLPipe(
        extractor,
        (unique_tces, prep_values),
        loader,
        bronze_dir,
        silver_dir,
    )
    pipe.run()
