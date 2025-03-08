# Copyright 2024-2025 Mohammed Nawabuddin
# SPDX-License-Identifier: Apache-2.0

from typing import Iterator

from astropy.io import fits
from jax import Array, numpy as jnp

from .utils import collect_dict


@collect_dict
def download(
    url: str,
    ext: int,
    data: list[str],
) -> Iterator[tuple[str, Array]]:
    with fits.open(
        url,
        mode="readonly",
        lazy_load_hdus=True,
        use_fsspec=True,
    ) as hdu_list:
        for d in data:
            array = hdu_list[ext].data[d]
            array = array.byteswap()
            array = array.view(array.dtype.newbyteorder())
            yield d, jnp.asarray(array)
