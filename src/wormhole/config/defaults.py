# Copyright 2024-2025 Mohammed Nawabuddin
# SPDX-License-Identifier: Apache-2.0

defaults = {
    "urls": {
        "bulk": {
            "host": "archive.stsci.edu",
        }
    },
    "async": {
        "batch_size": 100,
    },
    "data": {
        "catalogue": {
            "path": ["data"],
            "bronze": {
                "path": ["bronze"],
            },
            "silver": {
                "path": ["silver"],
            },
            "gold": {
                "path": ["gold"],
            },
            "lc": {
                "path": ["lc", "curves"],
            },
            "bulk_lc": {
                "path": ["bulk", "pixel_products", "lc"],
            },
            "bulk_tce": {
                "path": ["bulk", "catalogs", "tce"],
            },
            "lc_metadata": {
                "path": ["lc", "meta"],
            },
            "tce_metadata": {
                "path": ["tce", "meta"],
            },
            "metadataset": {
                "path": ["lc", "metadataset"],
            },
            "lc_manifest": {
                "path": ["lc", "manifest"],
            },
        },
    },
}
