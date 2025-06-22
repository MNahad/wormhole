# Copyright 2024-2025 Mohammed Nawabuddin
# SPDX-License-Identifier: Apache-2.0

from os import path

import optax

from wormhole.config import config
from wormhole.dataset import LightCurveDataLoader
from .model_gen import BasicRNNClassifier
from .trainer import train


def get_config() -> dict:
    conf = config()
    return {**conf["training"]}


def get_dataloader(
    manifest_dir: str,
    lc_dir: str,
    split_keys: tuple[str, ...],
    *,
    splits: dict,
    batch_size: int,
    num_epochs: int,
    allowed_labels_by_split: dict[str, tuple[bool, ...]],
) -> LightCurveDataLoader:
    allowed_label_split_indices = {
        i: allowed_labels_by_split[k]
        for i, k in enumerate(split_keys)
        if k in allowed_labels_by_split
    }
    loader = LightCurveDataLoader(
        manifest_dir,
        lc_dir,
        split_ratio=(splits[k] for k in split_keys),
        batch_size=batch_size,
        num_epochs=num_epochs,
        allowed_labels_split_indices=allowed_label_split_indices,
    )
    return loader


def main() -> None:
    conf = get_config()
    gold_dir = path.join(
        *(
            config()["data"]["catalogue"]["path"]
            + config()["data"]["catalogue"]["gold"]["path"]
        )
    )
    train_set, test_set, eval_set = get_dataloader(
        gold_dir,
        gold_dir,
        ("train", "test", "eval"),
        splits=conf["splits"],
        batch_size=conf["batch_size"],
        num_epochs=conf["num_epochs"],
        allowed_labels_by_split=conf["allowed_labels_by_split"],
    ).get()
    model = BasicRNNClassifier()
    for step, loss, _, _, _ in train(
        model=model,
        dataset=iter(train_set),
        tx=optax.adam(0.01),
    ):
        print(f"step: {step} loss: {loss}")
