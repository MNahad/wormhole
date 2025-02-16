# Copyright 2024-2025 Mohammed Nawabuddin
# SPDX-License-Identifier: Apache-2.0

import hashlib
import os
from os import path
from typing import Callable, Iterator, Optional
import unittest

from wormhole.config import config
from wormhole.pipeline import (
    process_metadataset,
    join_datasets,
    generate_lightcurve_manifest,
    process_lightcurves,
)
from . import synth


class PreprocessingPipelineTestCase(unittest.TestCase):

    @staticmethod
    def _list_dir(
        dir: str,
        predicate: Optional[Callable[[str], bool]] = None,
    ) -> Iterator[str]:
        predicate = (
            predicate
            if predicate
            else lambda file: path.isfile(path.join(dir, file))
        )
        return (file for file in os.listdir(dir) if predicate(file))

    @staticmethod
    def _get_file_hash(file: str) -> str:
        with open(file, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()

    @staticmethod
    def _list_hashes_in_dirs(
        baseline: str,
        target: str,
    ) -> Iterator[tuple[str, str, str]]:
        return (
            (
                file,
                PreprocessingPipelineTestCase._get_file_hash(
                    path.join(baseline, file)
                ),
                PreprocessingPipelineTestCase._get_file_hash(
                    path.join(target, file)
                ),
            )
            for file in PreprocessingPipelineTestCase._list_dir(baseline)
        )

    @classmethod
    def setUpClass(cls) -> None:
        synth.use_tmp_dir()

    @classmethod
    def tearDownClass(cls) -> None:
        synth.clear_tmp_dir()

    def setUp(self) -> None:
        conf = config()
        bronze = conf["data"]["catalogue"]["bronze"]["path"]
        silver = conf["data"]["catalogue"]["silver"]["path"]
        gold = conf["data"]["catalogue"]["gold"]["path"]
        tmp_dir = synth.get_tmp_dir()
        baseline_dir = synth.get_baseline_dir()
        self.bronze_dir = path.join(tmp_dir, *bronze)
        self.silver_dir = path.join(tmp_dir, *silver)
        self.gold_dir = path.join(tmp_dir, *gold)
        self.baseline_bronze_dir = path.join(baseline_dir, *bronze)
        self.baseline_silver_dir = path.join(baseline_dir, *silver)
        self.baseline_gold_dir = path.join(baseline_dir, *gold)

    def test_a_metadataset_pipe(self) -> None:
        process_metadataset(self.bronze_dir, self.silver_dir)
        for dir_path in (
            config()["data"]["catalogue"]["lc_metadata"]["path"],
            config()["data"]["catalogue"]["tce_metadata"]["path"],
        ):
            for hashes in PreprocessingPipelineTestCase._list_hashes_in_dirs(
                path.join(self.baseline_silver_dir, *dir_path),
                path.join(self.silver_dir, *dir_path),
            ):
                with self.subTest(file=".".join((*dir_path, hashes[0]))):
                    self.assertEqual(hashes[1], hashes[2])

    def test_b_join_datasets(self) -> None:
        join_datasets(self.silver_dir, self.gold_dir)
        dir_path = config()["data"]["catalogue"]["metadataset"]["path"]
        for hashes in PreprocessingPipelineTestCase._list_hashes_in_dirs(
            path.join(self.baseline_gold_dir, *dir_path),
            path.join(self.gold_dir, *dir_path),
        ):
            with self.subTest(file=".".join((*dir_path, hashes[0]))):
                self.assertEqual(hashes[1], hashes[2])

    def test_c_generate_manifest(self) -> None:
        generate_lightcurve_manifest(self.gold_dir, tce_ratio=(1, 1))
        dir_path = config()["data"]["catalogue"]["lc_manifest"]["path"]
        for hashes in PreprocessingPipelineTestCase._list_hashes_in_dirs(
            path.join(self.baseline_gold_dir, *dir_path),
            path.join(self.gold_dir, *dir_path),
        ):
            with self.subTest(file=".".join((*dir_path, hashes[0]))):
                self.assertEqual(hashes[1], hashes[2])

    def test_d_lightcurve_pipe(self) -> None:
        process_lightcurves(self.silver_dir, self.gold_dir)
        dir_path = config()["data"]["catalogue"]["lc"]["path"]
        baseline_root_dir = path.join(self.baseline_gold_dir, *dir_path)
        out_root_dir = path.join(self.gold_dir, *dir_path)
        for sub_dir in PreprocessingPipelineTestCase._list_dir(
            baseline_root_dir,
            lambda el: path.isdir(path.join(baseline_root_dir, el)),
        ):
            for hashes in PreprocessingPipelineTestCase._list_hashes_in_dirs(
                path.join(baseline_root_dir, sub_dir),
                path.join(out_root_dir, sub_dir),
            ):
                with self.subTest(
                    file=".".join((*dir_path, sub_dir, hashes[0]))
                ):
                    self.assertEqual(hashes[1], hashes[2])


if __name__ == "__main__":
    unittest.main()
