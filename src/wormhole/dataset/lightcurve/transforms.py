# Copyright 2024-2025 Mohammed Nawabuddin
# SPDX-License-Identifier: Apache-2.0

import grain.python as grain
from jax import Array, numpy as jnp

from . import model

type Pytree = tuple[Array, Array, Array]


class FilterLabels(grain.FilterTransform):
    tce_labels: tuple[bool, ...]

    def __init__(self, tce_labels: tuple[bool, ...]) -> None:
        super().__init__()
        self.tce_labels = tce_labels

    def filter(self, element: model.LightCurve) -> bool:
        return element.meta.has_tce in self.tce_labels


class PytreeTransform(grain.MapTransform):

    def map(self, element: model.LightCurve) -> model.Pytree:
        return element.to_pytree()


class StackTransform(grain.MapTransform):

    def map(self, element: model.Pytree) -> Pytree:
        return (
            jnp.column_stack(
                (
                    jnp.nan_to_num(element[0], nan=1e-6),
                    jnp.nan_to_num(element[1]),
                )
            ),
            element[2],
            element[3],
        )
