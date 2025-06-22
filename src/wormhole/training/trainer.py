# Copyright 2024-2025 Mohammed Nawabuddin
# SPDX-License-Identifier: Apache-2.0

from os import path
from typing import Iterator, Optional

import flax.core
from flax.training import train_state
from flax import linen as nn
import jax
from jax import Array
import optax

from wormhole.config import config
from wormhole.dataset import LightCurve
from .checkpointer import enable_checkpointing


@enable_checkpointing(path.join(*(config()["checkpoints"]["path"] + ("tmp",))))
def train(
    *,
    model: nn.Module,
    tx: optax.GradientTransformation,
    dataset: Iterator[LightCurve],
    step: int = 0,
    state: Optional[train_state.TrainState] = None,
    wirings_constants: Optional[flax.core.FrozenDict] = None,
) -> Iterator[
    tuple[
        int,
        Array,
        train_state.TrainState,
        Iterator[LightCurve],
        flax.core.FrozenDict,
    ]
]:
    rngs = {"params": jax.random.key(0)}
    for lcs in dataset:
        if step == 0:
            state, wirings_constants = _create_train_state_and_variables(
                model,
                lcs,
                rngs,
                tx,
            )
        state, loss = _train(state, lcs, wirings_constants, rngs)
        step += 1
        yield step, loss, state, dataset, wirings_constants


def _create_train_state_and_variables(
    model: nn.Module,
    lcs: LightCurve,
    rngs: dict[str, Array],
    tx: optax.GradientTransformation,
) -> tuple[train_state.TrainState, flax.core.FrozenDict]:
    variables = model.init(rngs, lcs[0], seq_lengths=lcs[2])
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
def _train(
    state: train_state.TrainState,
    lcs: LightCurve,
    wirings_constants: flax.core.FrozenDict,
    rngs: dict[str, Array],
) -> tuple[train_state.TrainState, Array]:

    def loss_fn(params):
        outputs = state.apply_fn(
            {"params": params, "wirings_constants": wirings_constants},
            lcs[0],
            rngs=rngs,
            seq_lengths=lcs[2],
        )
        return optax.sigmoid_binary_cross_entropy(
            outputs.reshape(lcs[0].shape[0]),
            lcs[1].reshape(lcs[0].shape[0]),
        ).mean()

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss
