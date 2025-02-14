# Copyright 2024 Mohammed Nawabuddin
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from os import path
from typing import Optional

from wormhole.common.fs import make_dir
from wormhole.common.http import Http


@dataclass
class ContextManager:
    lc_base_path_pattern: tuple[str, str] = field(
        default=(
            "/missions/tess/download_scripts/sector/",
            "tesscurl_sector_{}_lc.sh",
        )
    )
    lc_fast_path_pattern: tuple[str, str] = field(
        default=(
            "/missions/tess/download_scripts/sector/",
            "tesscurl_sector_{}_fast-lc.sh",
        )
    )
    lc_base_subdir: str = field(default="scripts/")
    lc_fast_subdir: str = field(default="fast_scripts/")
    lc_max_sector: int = field(default=26)
    lc_fast_sector_start: int = field(default=27)
    tce_base_path: str = field(default="/missions/tess/catalogs/tce")
    tce_base_subdir: str = field(default="csvs/")
    tce_files: list[str] = field(
        default_factory=lambda: [
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
    )


class STScI:
    lc_out_path: str
    tce_out_path: str
    client: Http
    context: ContextManager

    def __init__(
        self,
        out_dir: str,
        host: str = "archive.stsci.edu",
        context: Optional[ContextManager] = None,
    ) -> None:
        self.lc_out_path = path.abspath(path.join(out_dir, "lc/"))
        self.tce_out_path = path.abspath(path.join(out_dir, "tce/"))
        self.client = Http(host)
        self.context = context if context else ContextManager()

    def download(self) -> None:
        script_paths = [
            path.join(self.lc_out_path, dir)
            for dir in [
                self.context.lc_base_subdir,
                self.context.lc_fast_subdir,
            ]
        ]
        for script_path in script_paths:
            make_dir(script_path)
        lc_out_dir = path.join(self.lc_out_path, self.context.lc_base_subdir)
        lc_fast_out_dir = path.join(
            self.lc_out_path, self.context.lc_fast_subdir
        )
        for i in range(self.context.lc_max_sector):
            url_file = self.context.lc_base_path_pattern[1].format(i + 1)
            self.client.get(
                self.context.lc_base_path_pattern[0] + url_file,
                path.join(lc_out_dir, url_file),
            )
            if i >= self.context.lc_fast_sector_start:
                fast_url_file = self.context.lc_fast_path_pattern[1].format(
                    i + 1
                )
                self.client.get(
                    self.context.lc_fast_path_pattern[0] + fast_url_file,
                    path.join(lc_fast_out_dir, fast_url_file),
                )
        tce_out_dir = path.join(
            self.tce_out_path, self.context.tce_base_subdir
        )
        make_dir(tce_out_dir)
        for file in self.context.tce_files:
            self.client.get(
                path.join(self.context.tce_base_path, file),
                path.join(tce_out_dir, file),
            )
        self.client.close()
