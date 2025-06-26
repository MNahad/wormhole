# Copyright 2024-2025 Mohammed Nawabuddin
# SPDX-License-Identifier: Apache-2.0

from os import path
from typing import Optional

import jax
import optax

from wormhole.config import config
from .loader import get_dataloader, get_sample_dataloader
from .model_gen import get_models
from .trainer import create_train_state_and_constants, train


def run(job_id: Optional[str] = None) -> None:
    training_conf = config()["training"]
    gold_dir = path.join(
        *(
            config()["data"]["catalogue"]["path"]
            + config()["data"]["catalogue"]["gold"]["path"]
        )
    )
    train_dataset, test_dataset, _ = get_dataloader(
        gold_dir,
        gold_dir,
        ("train", "test", "eval"),
        splits=training_conf["splits"],
        batch_size=training_conf["batch_size"],
        num_epochs=training_conf["num_epochs"],
        allowed_labels_by_split=training_conf["allowed_labels_by_split"],
    ).get()
    sample_dataloader = get_sample_dataloader(
        gold_dir,
        gold_dir,
        batch_size=training_conf["batch_size"],
        allowed_labels_by_split=training_conf["allowed_labels_by_split"],
    ).get()[0]
    model = get_models()[training_conf["active_model"]]()
    rngs = {"params": jax.random.key(0)}
    train_state, wirings_constants = create_train_state_and_constants(
        model,
        rngs,
        optax.adam(training_conf["hyperparameters"]["adam"]),
        next(iter(sample_dataloader)),
    )
    print("STEP,LOSS")
    for (step, loss), _ in train(
        iter(train_dataset),
        rngs,
        train_state,
        wirings_constants,
        job_id,
    ):
        print(f"{step},{loss}")
