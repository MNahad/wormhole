# Copyright 2024-2025 Mohammed Nawabuddin
# SPDX-License-Identifier: Apache-2.0

import os

from .configuration import create_config

config_path = os.environ.get("WORMHOLE_CONFIG", None)

config = create_config(config_path)

__all__ = ["config"]
