# Copyright 2024-2025 Mohammed Nawabuddin
# SPDX-License-Identifier: Apache-2.0

from os import path
from typing import Iterator

import flax.core
from flax.training import train_state
from flax import linen as nn
from jax import Array

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
) -> Iterator[tuple[Array, Array]]:
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
        yield _sigmoid(logits) > 0.5, lcs[1]


def _sigmoid(logits: Array) -> Array:
    return nn.sigmoid(logits)
