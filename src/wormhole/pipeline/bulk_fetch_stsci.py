# Copyright 2024 Mohammed Nawabuddin
# SPDX-License-Identifier: Apache-2.0

from wormhole.connectors.stsci.bulk_fetcher import STScI


def fetch_bulk_data(bronze_dir: str) -> None:
    stsci_client = STScI(bronze_dir)
    stsci_client.download()
