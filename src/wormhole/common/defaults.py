# Copyright 2024-2025 Mohammed Nawabuddin
# SPDX-License-Identifier: Apache-2.0

import pyarrow as pa


def metadataset_schema() -> dict[str, pa.DataType]:
    return {
        "sector": pa.uint8(),
        "ticid": pa.uint64(),
        "url": pa.string(),
        "has_tce": pa.bool_(),
    }


def lightcurve_data_keys() -> list[str]:
    return ["TIME", "PDCSAP_FLUX", "QUALITY"]


def lightcurve_array_size() -> int:
    return 32_768
