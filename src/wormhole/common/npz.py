# Copyright 2024-2025 Mohammed Nawabuddin
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

from jax import Array, numpy as jnp

type Npz = dict[str, Array]


def read(file: str, data: Optional[list[str]] = None) -> Npz:
    with jnp.load(file, allow_pickle=False) as npz_file:
        return {
            k: jnp.asarray(npz_file[k]) for k in (data if data else npz_file)
        }


def write(file: str, npz: Npz) -> None:
    jnp.savez(file, allow_pickle=False, **npz)
