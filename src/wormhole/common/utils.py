# Copyright 2024-2025 Mohammed Nawabuddin
# SPDX-License-Identifier: Apache-2.0

from functools import reduce, wraps
from itertools import filterfalse
import re
from typing import Callable, Iterable, Iterator


def chain[T, U, V](
    f_initial: Callable[[T], U],
    *f_inters: Callable[[U], U],
    f_final: Callable[[U], V],
) -> Callable[[T], V]:
    return lambda arg: (
        f_final(
            reduce(
                lambda carry, f: f(carry),
                f_inters,
                f_initial(arg),
            )
        )
    )


def match_substring(
    iter: Iterable[str],
    pattern: re.Pattern,
) -> Iterator[re.Match[str]]:
    for element in iter:
        match = re.search(pattern, element)
        if match:
            yield match


def filter_no_prefix(iter: Iterable[str], prefix: str) -> Iterator[str]:
    return filterfalse(lambda x: x.startswith(prefix), iter)


def unique(
    iter: Iterable[dict[str, str]],
    key: str,
) -> Iterator[dict[str, str]]:
    uniques = set()
    for element in iter:
        if element[key] in uniques:
            continue
        uniques.add(element[key])
        yield element


def get_vertex_pairs_from_edges(
    edges: tuple[float, ...],
) -> tuple[tuple[float, float], ...]:
    pairs = []
    for i, edge in enumerate(edges):
        if i == 0:
            pairs.append((0.0, edge))
        else:
            pairs.append((vertex := pairs[i - 1][-1], vertex + edge))
    return tuple(pairs)


def collect_dict[**P, Q](
    fn: Callable[P, Iterator[tuple[str, Q]]],
) -> Callable[P, dict[str, Q]]:

    @wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> dict[str, Q]:
        return {k: v for k, v in fn(*args, **kwargs)}

    return wrapper


def iterate[T](fn: Callable[[T], T]) -> Callable[[Iterable[T]], Iterator[T]]:

    @wraps(fn)
    def wrapper(it: Iterable[T]) -> Iterator[T]:
        return (fn(t) for t in it)

    return wrapper
