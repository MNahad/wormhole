# Copyright 2024-2025 Mohammed Nawabuddin
# SPDX-License-Identifier: Apache-2.0

from os import path
import tomllib
from types import MappingProxyType
from typing import Callable

from .defaults import defaults


def create_config() -> Callable[[], MappingProxyType]:
    config = _deep_freeze(_use_defaults(_load_from_file(), defaults))

    def get_config() -> MappingProxyType:
        return config

    return get_config


def _load_from_file(file: str = "wormhole.toml") -> dict:
    if not path.exists(file):
        return dict()
    with open(file, "rb") as f:
        return tomllib.load(f)


def _use_defaults(conf: dict, default: dict) -> dict:
    for k, v in default.items():
        if k in conf:
            if isinstance(v, dict):
                default[k] = _use_defaults(conf[k], default[k])
            else:
                default[k] = conf[k]
    return default


def _deep_freeze(conf: dict) -> MappingProxyType:
    for k, v in conf.items():
        if isinstance(v, dict):
            conf[k] = _deep_freeze(v)
        elif isinstance(v, list):
            conf[k] = tuple(v)
    return MappingProxyType(conf)
