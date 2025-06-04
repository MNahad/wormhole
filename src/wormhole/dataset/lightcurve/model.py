# Copyright 2024-2025 Mohammed Nawabuddin
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field, InitVar

from jax import Array, numpy as jnp

import wormhole.common.npz as npz

type Pytree = tuple[Array, Array, Array, Array]


@dataclass
class Data:
    data: InitVar[npz.Npz]
    time: Array = field(init=False)
    pdcsap_flux: Array = field(init=False)
    quality: Array = field(init=False)
    sequence_length: int = field(init=False)

    def __post_init__(self, data: npz.Npz) -> None:
        self.time = data["TIME"]
        self.pdcsap_flux = data["PDCSAP_FLUX"]
        self.quality = data["QUALITY"]
        self.sequence_length = int(jnp.argmin(self.time))


@dataclass(frozen=True)
class Meta:
    sector: str
    ticid: str
    has_tce: bool


@dataclass(frozen=True)
class LightCurve:
    data: Data
    meta: Meta

    def to_pytree(self) -> Pytree:
        return (
            self.data.time,
            self.data.pdcsap_flux,
            jnp.array(self.meta.has_tce, dtype=jnp.bool),
            jnp.array(self.data.sequence_length, dtype=jnp.uint16),
        )
