# Copyright 2024-2025 Mohammed Nawabuddin
# SPDX-License-Identifier: Apache-2.0

from wormhole.connectors.stsci.bulk_fetcher import STScI


def fetch_bulk_data(bronze_dir: str, **kwargs) -> None:
    stsci_client = STScI(bronze_dir)
    stsci_client.download()
