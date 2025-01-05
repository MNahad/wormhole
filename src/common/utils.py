# Copyright 2024 Mohammed Nawabuddin
# SPDX-License-Identifier: Apache-2.0

from functools import reduce
from itertools import filterfalse
import re
from typing import Callable, Iterable, Iterator


def chain[
    T,
    U,
    V,
](
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
