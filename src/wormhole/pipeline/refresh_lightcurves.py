# Copyright 2024 Mohammed Nawabuddin
# SPDX-License-Identifier: Apache-2.0

from wormhole.connectors.stsci.lc_store import LightCurveStore


def refresh_lightcurve_store(
    silver_dir: str,
    gold_dir: str,
    /,
    debug=False,
) -> None:
    lc_store = LightCurveStore(gold_dir, silver_dir)
    lc_store.download(debug=debug)
