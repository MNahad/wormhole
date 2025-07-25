# Copyright 2024-2025 Mohammed Nawabuddin
# SPDX-License-Identifier: Apache-2.0


def defaults() -> dict:
    return {
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
        "pipelines": {
            "args": {
                "debug": False,
                "tce_sample_ratio": [20_000, 20_000],
            },
        },
        "training": {
            "splits": {
                "train": 0.5,
                "validation": 0.3,
                "test": 0.2,
            },
            "allowed_labels_by_split": {
                "train": [True, False],
                "validation": [True, False],
                "test": [True, False],
            },
            "batch_size": 10,
            "num_epochs": 10,
            "hyperparameters": {
                "adam": 0.01,
                "models": {
                    "basic": {
                        "units": 11,
                    }
                },
            },
            "active_model": "basic",
        },
        "checkpoints": {
            "path": ["checkpoints"],
            "save_interval_steps": 100,
        },
    }
