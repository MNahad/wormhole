# Copyright 2024-2025 Mohammed Nawabuddin
# SPDX-License-Identifier: Apache-2.0

from os import path
from typing import Optional

import flax.core
from flax.training import train_state
import jax
from jax import Array, numpy as jnp
import optax

from wormhole.config import config
from wormhole.dataset import LightCurve
from .evaluator import eval, get_metrics
from .loader import get_dataloader, get_sample_dataloader
from .model_gen import get_models
from .trainer import create_train_state_and_constants, train


def run(
    job_id: Optional[str] = None,
    *,
    mode: Optional[str] = "train",
) -> None:
    _train(job_id) if mode == "train" else _eval(job_id if job_id else "")


def _train(job_id: Optional[str] = None) -> None:
    (rngs, (train_dataset, _, _), train_state, wirings_constants) = _prep()
    print("STEP,LOSS")
    for (step, loss), _ in train(
        rngs,
        iter(train_dataset),
        train_state,
        wirings_constants,
        job_id,
    ):
        print(f"{step},{loss}")


def _eval(job_id: str) -> None:
    (rngs, (_, _, test_dataset), train_state, wirings_constants) = _prep()
    tps, tns, fps, fns = jnp.array(0), jnp.array(0), jnp.array(0), jnp.array(0)
    step = 0
    print(
        "STEP,TRUE_POSITIVE,TRUE_NEGATIVE,FALSE_POSITIVE,FALSE_NEGATIVE,ACCURACY,PRECISION,RECALL,FALSEALARM"
    )
    for _, (tp, tn, fp, fn) in eval(
        rngs,
        iter(test_dataset),
        train_state,
        wirings_constants,
        job_id,
    ):
        tps += tp
        tns += tn
        fps += fp
        fns += fn
        step += 1
        accuracy, precision, recall, false_alarm = get_metrics(
            tps,
            tns,
            fps,
            fns,
        )
        print(
            f"{step},{tps},{tns},{fps},{fns},{accuracy},{precision},{recall},{false_alarm}"
        )


def _prep() -> tuple[
    dict[str, Array],
    tuple[LightCurve, ...],
    train_state.TrainState,
    flax.core.FrozenDict,
]:
    training_conf = config()["training"]
    gold_dir = path.join(
        *(
            config()["data"]["catalogue"]["path"]
            + config()["data"]["catalogue"]["gold"]["path"]
        )
    )
    datasets = get_dataloader(
        gold_dir,
        gold_dir,
        training_conf["splits"].keys(),
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
    return (rngs, datasets, train_state, wirings_constants)
