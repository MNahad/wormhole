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
def equalise_size(lc: LC) -> LC:
    seq_len = len(lc["TIME"])
    for key in defaults.lightcurve_data_keys():
        lc[key] = jnp.pad(
            lc[key],
            (0, defaults.lightcurve_array_size() - seq_len),
            mode="constant",
            constant_values=(_get_array_fill_value(key),),
        )
    return lc


@iterate
@jit
def filter_low_q(lc: LC) -> LC:
    quality = lc["QUALITY"]
    low_mask = _generate_low_q_mask()
    i_bad = _jnp_nonzero(quality & low_mask)[0]
    lc["PDCSAP_FLUX"] = lc["PDCSAP_FLUX"].at[i_bad].set(jnp.nan)
    return lc


@iterate
@jit
def standardise_t(lc: LC) -> LC:
    lc["TIME"] = (lc["TIME"] - lc["TIME"][0]) * 1_440
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
    i_not_nan = _jnp_nonzero(~jnp.isnan(pdcsap_flux))[0]
    for key in defaults.lightcurve_data_keys():
        lc[key] = lc[key].at[i_not_nan].get()
    return lc


@iterate
@jit
def use_delta_t(lc: LC) -> LC:
    t = lc["TIME"]
    dt = t - jnp.roll(t, 1)
    dt = dt.at[0].set(1e-6)
    lc["TIME"] = dt
    return lc


@jit
def _generate_low_q_mask() -> int:
    low_q_flags = [1, 2, 3, 4, 5, 6, 8, 10, 13, 15]
    mask = 0
    for b in low_q_flags:
        mask += 1 << (b - 1)
    return mask


def _get_array_fill_value(key: str) -> float:
    match key:
        case "TIME":
            return jnp.nan
        case "PDCSAP_FLUX":
            return jnp.nan
        case "QUALITY":
            return 0
        case _:
            return -1


_jnp_nonzero = partial(
    jit(jnp.nonzero, static_argnames=("size",)),
    size=defaults.lightcurve_array_size(),
    fill_value=defaults.lightcurve_array_size(),
)
