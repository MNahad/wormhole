# Copyright 2024-2025 Mohammed Nawabuddin
# SPDX-License-Identifier: Apache-2.0

from wormhole.config import config
from wormhole.connectors.stsci.lc_store import LightCurveStore


def recache_lightcurve_store(silver_dir: str, gold_dir: str, **kwargs) -> None:
    debug = config()["pipelines"]["args"]["debug"]
    lc_store = LightCurveStore(gold_dir, silver_dir)
    lc_store.download(debug=debug)
