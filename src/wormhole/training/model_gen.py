# Copyright 2024-2025 Mohammed Nawabuddin
# SPDX-License-Identifier: Apache-2.0

from flax import linen as nn
from flaxoil import ltc
from jax import Array, numpy as jnp


class BasicRNNClassifier(nn.Module):
    @nn.compact
    def __call__(self, inputs: Array, *, seq_lengths: Array) -> Array:
        x = nn.RNN(
            ltc.LTCCell(
                {"ncp": {"units": 11, "output_size": 1}},
                irregular_time_mode=True,
            ),
            variable_broadcast=["params", "wirings_constants"],
        )(inputs)
        x = x.at[
            jnp.arange(x.shape[0]).reshape(x.shape[0], 1),
            (seq_lengths - 1).reshape(x.shape[0], 1),
            :,
        ].get()
        return nn.Dense(1)(x)
