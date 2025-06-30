# Copyright 2024-2025 Mohammed Nawabuddin
# SPDX-License-Identifier: Apache-2.0

from os import path
from typing import Iterator

import flax.core
from flax.training import train_state
from flax import linen as nn
import jax
from jax import Array, numpy as jnp

from wormhole.config import config
from wormhole.dataset import LightCurve
from .checkpointer import restore_train_state_and_constants
from .trainer import loss_fn


def eval(
    rngs: dict[str, Array],
    dataset: Iterator[LightCurve],
    initial_state: train_state.TrainState,
    initial_wirings_constants: flax.core.FrozenDict,
    job_id: str,
) -> Iterator[tuple[tuple[Array, Array], tuple[Array, Array, Array, Array]]]:
    restored = restore_train_state_and_constants(
        path.join(*config()["checkpoints"]["path"]),
        job_id,
        initial_wirings_constants,
        initial_state,
    )
    state = restored[0]
    wirings_constants = restored[1]
    for lcs in dataset:
        (_, logits) = loss_fn(
            state.params,
            state,
            lcs,
            wirings_constants,
            rngs,
        )
        prediction = nn.sigmoid(logits) > 0.5
        yield (prediction, lcs[1]), _get_confusion_matrix(prediction, lcs[1])


@jax.jit
def get_metrics(
    tp: Array,
    tn: Array,
    fp: Array,
    fn: Array,
) -> tuple[Array, Array, Array, Array]:
    return (
        (tp + tn) / (tp + tn + fp + fn),
        tp / (tp + fp),
        tp / (tp + fn),
        fp / (fp + tn),
    )


@jax.jit
def _get_confusion_matrix(
    prediction: Array,
    truth: Array,
) -> tuple[Array, Array, Array, Array]:
    not_prediction = ~prediction
    not_truth = ~truth
    return (
        jnp.sum(prediction & truth),
        jnp.sum(not_prediction & not_truth),
        jnp.sum(prediction & not_truth),
        jnp.sum(not_prediction & truth),
    )
