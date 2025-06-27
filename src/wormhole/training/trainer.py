# Copyright 2024-2025 Mohammed Nawabuddin
# SPDX-License-Identifier: Apache-2.0

from os import path
from typing import Iterator

import flax.core
from flax.training import train_state
from flax import linen as nn
import jax
from jax import Array
import optax
import orbax.checkpoint as ocp

from wormhole.config import config
from wormhole.dataset import LightCurve
from .checkpointer import enable_checkpointing


@enable_checkpointing(
    path.join(*config()["checkpoints"]["path"]),
    ocp.CheckpointManagerOptions(
        save_interval_steps=config()["checkpoints"]["save_interval_steps"]
    ),
)
def train(
    rngs: dict[str, Array],
    dataset: Iterator[LightCurve],
    state: train_state.TrainState,
    wirings_constants: flax.core.FrozenDict,
    step: int,
) -> Iterator[
    tuple[
        tuple[int, Array],
        tuple[Iterator[LightCurve], train_state.TrainState],
    ]
]:
    for lcs in dataset:
        state, loss = _train(state, lcs, wirings_constants, rngs)
        step += 1
        yield (step, loss), (dataset, state)


def create_train_state_and_constants(
    model: nn.Module,
    rngs: dict[str, Array],
    tx: optax.GradientTransformation,
    initial_lcs: LightCurve,
) -> tuple[train_state.TrainState, flax.core.FrozenDict]:
    variables = model.init(rngs, initial_lcs[0], seq_lengths=initial_lcs[2])
    params = variables["params"]
    wirings_constants = variables["wirings_constants"]
    return (
        train_state.TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=tx,
        ),
        wirings_constants,
    )


@jax.jit
def loss_fn(
    params: flax.core.FrozenDict,
    state: train_state.TrainState,
    lcs: LightCurve,
    wirings_constants: flax.core.FrozenDict,
    rngs: dict[str, Array],
) -> tuple[Array, Array]:
    logits = state.apply_fn(
        {"params": params, "wirings_constants": wirings_constants},
        lcs[0],
        rngs=rngs,
        seq_lengths=lcs[2],
    )
    logits_reshaped = logits.reshape(lcs[0].shape[0])
    labels_reshaped = lcs[1].reshape(lcs[0].shape[0])
    loss = optax.sigmoid_binary_cross_entropy(
        logits_reshaped,
        labels_reshaped,
    ).mean()
    return loss, logits_reshaped


@jax.jit
def _train(
    state: train_state.TrainState,
    lcs: LightCurve,
    wirings_constants: flax.core.FrozenDict,
    rngs: dict[str, Array],
) -> tuple[train_state.TrainState, Array]:
    grad_fn = jax.value_and_grad(loss_fn, 0, has_aux=True)
    (loss, _), grads = grad_fn(
        state.params,
        state,
        lcs,
        wirings_constants,
        rngs,
    )
    state = state.apply_gradients(grads=grads)
    return state, loss
