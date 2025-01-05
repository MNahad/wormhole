# Copyright 2024 Mohammed Nawabuddin
# SPDX-License-Identifier: Apache-2.0

from os import path
import re

from common.fs import iter_files, make_dir
from common.http import Http
from common.utils import match_substring


class STScI:
    lc_out_path: str
    tce_out_path: str
    client: Http
    lc_base_path_pattern: tuple[str, str] = (
        "/missions/tess/download_scripts/sector/",
        "tesscurl_sector_{}_lc.sh",
    )
    lc_fast_path_pattern: tuple[str, str] = (
        "/missions/tess/download_scripts/sector/",
        "tesscurl_sector_{}_fast-lc.sh",
    )
    lc_base_subdir: str = "scripts/"
    lc_fast_subdir: str = "fast_scripts/"
    lc_max_sector: int = 26
    lc_fast_sector_start: int = 27
    tce_base_path: str = "/missions/tess/catalogs/tce"
    tce_base_subdir: str = "csvs/"
    tce_files: list[str] = [
        "tess2018206190142-s0001-s0001_dvr-tcestats.csv",
        "tess2018235142541-s0002-s0002_dvr-tcestats.csv",
        "tess2018263124740-s0003-s0003_dvr-tcestats.csv",
        "tess2018292093539-s0004-s0004_dvr-tcestats.csv",
        "tess2018319112538-s0005-s0005_dvr-tcestats.csv",
        "tess2018349182737-s0006-s0006_dvr-tcestats.csv",
        "tess2019008025936-s0007-s0007_dvr-tcestats.csv",
        "tess2019033200935-s0008-s0008_dvr-tcestats.csv",
        "tess2019059170935-s0009-s0009_dvr-tcestats.csv",
        "tess2019085221934-s0010-s0010_dvr-tcestats.csv",
        "tess2019113062933-s0011-s0011_dvr-tcestats.csv",
        "tess2019141104532-s0012-s0012_dvr-tcestats.csv",
        "tess2019170095531-s0013-s0013_dvr-tcestats.csv",
        "tess2019199201929-s0014-s0014_dvr-tcestats.csv",
        "tess2019227203528-s0015-s0015_dvr-tcestats.csv",
        "tess2019255032927-s0016-s0016_dvr-tcestats.csv",
        "tess2019281041526-s0017-s0017_dvr-tcestats.csv",
        "tess2019307033525-s0018-s0018_dvr-tcestats.csv",
        "tess2019332134924-s0019-s0019_dvr-tcestats.csv",
        "tess2019358235523-s0020-s0020_dvr-tcestats.csv",
        "tess2020021221522-s0021-s0021_dvr-tcestats.csv",
        "tess2020050191121-s0022-s0022_dvr-tcestats.csv",
        "tess2020079142120-s0023-s0023_dvr-tcestats.csv",
        "tess2020107065519-s0024-s0024_dvr-tcestats.csv",
        "tess2020135030118-s0025-s0025_dvr-tcestats.csv",
        "tess2020161181517-s0026-s0026_dvr-tcestats.csv",
    ]

    def __init__(
        self,
        bronze_dir: str,
        host: str = "archive.stsci.edu",
    ) -> None:
        self.lc_out_path = path.abspath(path.join(bronze_dir, "lc/"))
        self.tce_out_path = path.abspath(path.join(bronze_dir, "tce/"))
        self.client = Http(host)

    def download(self) -> None:
        script_paths = [
            path.join(self.lc_out_path, dir)
            for dir in [self.lc_base_subdir, self.lc_fast_subdir]
        ]
        for script_path in script_paths:
            make_dir(script_path)
        lc_out_dir = path.join(self.lc_out_path, self.lc_base_subdir)
        lc_fast_out_dir = path.join(self.lc_out_path, self.lc_fast_subdir)
        for i in range(self.lc_max_sector):
            url_file = self.lc_base_path_pattern[1].format(i + 1)
            self.client.get(
                self.lc_base_path_pattern[0] + url_file,
                path.join(lc_out_dir, url_file),
            )
            if i >= self.lc_fast_sector_start:
                fast_url_file = self.lc_fast_path_pattern[1].format(i + 1)
                self.client.get(
                    self.lc_fast_path_pattern[0] + fast_url_file,
                    path.join(lc_fast_out_dir, fast_url_file),
                )
        make_dir(path.join(self.tce_out_path, self.tce_base_subdir))
        tce_out_dir = path.join(self.tce_out_path, self.tce_base_subdir)
        for file in self.tce_files:
            self.client.get(
                path.join(self.tce_base_path, file),
                path.join(tce_out_dir, file),
            )
        self.client.close()


class FITS:
    client: Http
    script_dir: str
    fits_dir: str
    base_subdir: str = "scripts/"
    fast_subdir: str = "fast_scripts/"

    def __init__(
        self,
        bronze_dir: str,
        host: str = "mast.stsci.edu",
    ) -> None:
        self.script_dir = path.abspath(path.join(bronze_dir, "lc/"))
        self.fits_dir = path.join(self.script_dir, "fits/")
        self.client = Http(host)

    def download(self) -> None:
        make_dir(self.fits_dir)
        for dir in [self.base_subdir, self.fast_subdir]:
            for file in iter_files(path.join(self.script_dir, dir)):
                with open(file, "r") as in_f:
                    for url in match_substring(
                        in_f,
                        re.compile(r"(tess.+)\s(https://.+\.fits)$"),
                    ):
                        self.client.get(
                            url.group(2),
                            path.join(self.fits_dir, url.group(1)),
                        )
        self.client.close()
