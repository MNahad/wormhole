# Copyright 2024 Mohammed Nawabuddin
# SPDX-License-Identifier: Apache-2.0

from connectors.stsci.lc_store import LightCurveStore


def refresh_lc_store(bronze_dir: str, silver_dir: str, /, debug=False) -> None:
    lc_store = LightCurveStore(bronze_dir, silver_dir)
    lc_store.download(debug=debug)
