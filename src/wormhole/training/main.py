# Copyright 2024-2025 Mohammed Nawabuddin
# SPDX-License-Identifier: Apache-2.0

from os import path

import jax
import optax

from wormhole.config import config
from wormhole.dataset import LightCurve
from .loader import get_dataloader, get_sample_dataloader
from .model_gen import BasicRNNClassifier
from .trainer import create_train_state_and_constants, train


def get_config() -> dict:
    conf = config()
    return {**conf["training"]}


def main() -> None:
    conf = get_config()
    gold_dir = path.join(
        *(
            config()["data"]["catalogue"]["path"]
            + config()["data"]["catalogue"]["gold"]["path"]
        )
    )
    train_dataset, test_dataset, eval_dataset = get_dataloader(
        gold_dir,
        gold_dir,
        ("train", "test", "eval"),
        splits=conf["splits"],
        batch_size=conf["batch_size"],
        num_epochs=conf["num_epochs"],
        allowed_labels_by_split=conf["allowed_labels_by_split"],
    ).get()
    sample_dataloader = get_sample_dataloader(
        gold_dir,
        gold_dir,
        batch_size=conf["batch_size"],
        allowed_labels_by_split=conf["allowed_labels_by_split"],
    ).get()[0]

    def _get_sample_lcs() -> LightCurve:
        return next(iter(sample_dataloader))

    model = BasicRNNClassifier()
    rngs = {"params": jax.random.key(0)}
    train_state, wirings_constants = create_train_state_and_constants(
        model,
        rngs,
        optax.adam(0.01),
        _get_sample_lcs,
    )
    for step, loss, _, _, _ in train(
        dataset=iter(train_dataset),
        rngs=rngs,
        state=train_state,
        wirings_constants=wirings_constants,
    ):
        print(f"step: {step} loss: {loss}")
