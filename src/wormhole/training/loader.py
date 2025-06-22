# Copyright 2024-2025 Mohammed Nawabuddin
# SPDX-License-Identifier: Apache-2.0

from wormhole.dataset import LightCurveDataLoader


def get_dataloader(
    manifest_dir: str,
    lc_dir: str,
    split_keys: tuple[str, ...],
    *,
    splits: dict[str, float],
    batch_size: int,
    num_epochs: int,
    allowed_labels_by_split: dict[str, tuple[bool, ...]],
) -> LightCurveDataLoader:
    allowed_label_split_indices = {
        i: allowed_labels_by_split[k]
        for i, k in enumerate(split_keys)
        if k in allowed_labels_by_split
    }
    return LightCurveDataLoader(
        manifest_dir,
        lc_dir,
        split_ratio=(splits[k] for k in split_keys),
        batch_size=batch_size,
        num_epochs=num_epochs,
        allowed_labels_split_indices=allowed_label_split_indices,
    )


def get_sample_dataloader(
    manifest_dir: str,
    lc_dir: str,
    *,
    batch_size: int,
    allowed_labels_by_split: dict[str, tuple[bool, ...]],
) -> LightCurveDataLoader:
    return get_dataloader(
        manifest_dir,
        lc_dir,
        ("sample",),
        splits={"sample": 0.001},
        batch_size=batch_size,
        num_epochs=1,
        allowed_labels_by_split=allowed_labels_by_split,
    )
