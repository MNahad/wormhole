# Copyright 2024-2025 Mohammed Nawabuddin
# SPDX-License-Identifier: Apache-2.0

from functools import partial
from os import path
from typing import Iterable

from jax import Array, jit, numpy as jnp

import wormhole.common.defaults as defaults
import wormhole.common.npz as npz
from wormhole.common.utils import iterate

type LC = dict[str, Array]
type Pipe = Iterable[LC]


def extract(file: str, source: str) -> Pipe:
    lc = npz.read(path.join(source, file), defaults.lightcurve_data_keys())
    return (lc for _ in range(1))


def load(lcs: Pipe, file: str, sink: str) -> None:
    for lc in lcs:
        npz.write(path.join(sink, file), lc)


@iterate
@jit
def filter_low_q(lc: LC) -> LC:
    quality = lc["QUALITY"]
    low_mask = _generate_low_q_mask()
    i_bad = _jnp_nonzero(quality & low_mask)[0].at[: len(quality)].get()
    lc["PDCSAP_FLUX"] = lc["PDCSAP_FLUX"].at[i_bad].set(jnp.nan)
    return lc


@iterate
@jit
def standardise_t(lc: LC) -> LC:
    lc["TIME"] = lc["TIME"] - lc["TIME"][0]
    return lc


@iterate
@jit
def standardise_flux(lc: LC) -> LC:
    flux = lc["PDCSAP_FLUX"]
    mean = jnp.nanmean(flux)
    std = jnp.nanstd(flux)
    lc["PDCSAP_FLUX"] = (flux - mean) / std
    return lc


@iterate
@jit
def filter_nan(lc: LC) -> LC:
    pdcsap_flux = lc["PDCSAP_FLUX"]
    i_not_nan = (
        _jnp_nonzero(~jnp.isnan(pdcsap_flux))[0].at[: len(pdcsap_flux)].get()
    )
    for arr in defaults.lightcurve_data_keys():
        lc[arr] = lc[arr].at[i_not_nan].get()
    return lc


_jnp_nonzero = partial(
    jit(jnp.nonzero, static_argnames="size"),
    size=100_000,
    fill_value=-1,
)


@jit
def _generate_low_q_mask() -> int:
    low_q_flags = [1, 2, 3, 4, 5, 6, 8, 10, 13, 15]
    mask = 0
    for b in low_q_flags:
        mask += 1 << (b - 1)
    return mask
