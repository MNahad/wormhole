# Copyright 2024-2025 Mohammed Nawabuddin
# SPDX-License-Identifier: Apache-2.0

from os import path
from typing import Iterator

import optax

from wormhole.config import config
from wormhole.dataset import LightCurveDataLoader, LightCurve
from wormhole.training.model_gen import BasicRNNClassifier
from wormhole.training.trainer import train


def get_dataset(
    gold_dir: str,
    split_keys: tuple[str, ...],
    *,
    splits: dict,
    batch_size: int,
    num_epochs: int,
    allowed_labels_by_split: dict[str, tuple[bool, ...]],
) -> tuple[Iterator[LightCurve], ...]:
    allowed_label_split_indices = {
        i: allowed_labels_by_split[k]
        for i, k in enumerate(split_keys)
        if k in allowed_labels_by_split
    }
    sets = LightCurveDataLoader(
        gold_dir,
        gold_dir,
        split_ratio=(splits[k] for k in split_keys),
        batch_size=batch_size,
        num_epochs=num_epochs,
        allowed_labels_split_indices=allowed_label_split_indices,
    ).get()
    return sets


def main() -> None:
    conf = config()
    catalogue_path = conf["data"]["catalogue"]["path"]
    gold_dir = path.join(
        *(catalogue_path + conf["data"]["catalogue"]["gold"]["path"])
    )
    splits = conf["training"]["splits"]
    batch_size = conf["training"]["batch_size"]
    num_epochs = conf["training"]["num_epochs"]
    allowed_labels_by_split = conf["training"]["allowed_labels_by_split"]
    train_set, test_set, eval_set = get_dataset(
        gold_dir,
        ("train", "test", "eval"),
        splits=splits,
        batch_size=batch_size,
        num_epochs=num_epochs,
        allowed_labels_by_split=allowed_labels_by_split,
    )
    model = BasicRNNClassifier()
    for loss in train(model, train_set, optax.adam(0.01)):
        print(f"loss: {loss}")
