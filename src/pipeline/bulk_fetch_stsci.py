# Copyright 2024 Mohammed Nawabuddin
# SPDX-License-Identifier: Apache-2.0

from connectors.stsci.bulk_fetcher import STScI


def bulk_fetch(bronze_dir: str) -> None:
    stsci_client = STScI(bronze_dir)
    stsci_client.download()
