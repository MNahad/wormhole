# Copyright 2024 Mohammed Nawabuddin
# SPDX-License-Identifier: Apache-2.0

import asyncio
from typing import Callable, Iterable, Iterator


def exec[
    T, **P, Q
](fn: Callable[P, Q], iterable: Iterable[T], *fn_args) -> Iterator[Q]:
    coro = _exec_coro(fn, iterable, *fn_args)
    yield from asyncio.run(coro)


async def _exec_coro[
    T, **P, Q
](fn: Callable[P, Q], iterable: Iterable[T], *fn_args) -> list[Q]:
    loop = asyncio.get_running_loop()
    futures = (
        loop.run_in_executor(None, fn, *(it, *fn_args)) for it in iterable
    )
    return await asyncio.gather(*futures)
