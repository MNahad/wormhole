# Copyright 2024-2025 Mohammed Nawabuddin
# SPDX-License-Identifier: Apache-2.0

from functools import wraps
from os import path
from typing import Callable, Iterator, Mapping, Optional

import flax.core
from flax.training import train_state
import grain.python as grain
from jax import Array
import orbax.checkpoint as ocp

from wormhole.dataset import LightCurve

type _ReturnType = Iterator[
    tuple[
        int,
        Array,
        train_state.TrainState,
        Iterator[LightCurve],
        flax.core.FrozenDict,
    ]
]


def enable_checkpointing[**P](
    dir: str,
    options: Optional[ocp.CheckpointManagerOptions] = None,
) -> Callable[[Callable[P, _ReturnType]], Callable[P, _ReturnType]]:

    def checkpointer(fn: Callable[P, _ReturnType]) -> Callable[P, _ReturnType]:

        @wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> _ReturnType:
            with ocp.CheckpointManager(
                path.abspath(dir),
                options=options if options else ocp.CheckpointManagerOptions(),
            ) as mngr:
                step = 0
                state = None
                wirings_constants = None
                dataset = kwargs["dataset"]
                restored = _restore(mngr, dataset)
                if restored is not None:
                    step = restored[0]
                    restored_composite = restored[1]
                    dataset = restored_composite["loader"]
                    state = restored_composite["state"]
                    wirings_constants = restored_composite["wirings_constants"]
                kwargs["step"] = step
                kwargs["dataset"] = dataset
                kwargs["state"] = state
                kwargs["wirings_constants"] = wirings_constants
            for next_iter in fn(*args, **kwargs):
                _save(
                    mngr,
                    next_iter[0],
                    next_iter[2],
                    next_iter[3],
                    next_iter[4],
                )
                yield next_iter

        return wrapper

    return checkpointer


def _save(
    mngr: ocp.CheckpointManager,
    step: int,
    state: train_state.TrainState,
    loader_iter: grain.PyGrainDatasetIterator,
    wirings_constants: flax.core.FrozenDict,
) -> None:
    mngr.save(
        step,
        args=ocp.args.Composite(
            state=ocp.args.PyTreeSave(state),
            loader=grain.PyGrainCheckpointSave(loader_iter),
            wirings_constants=ocp.args.PyTreeSave(wirings_constants),
        ),
    )
    mngr.wait_until_finished()


def _restore(
    mngr: ocp.CheckpointManager,
    loader_iter: grain.PyGrainDatasetIterator,
) -> Optional[tuple[int, Mapping]]:
    latest_step = mngr.latest_step()
    return (
        (
            latest_step,
            mngr.restore(
                latest_step,
                args=ocp.args.Composite(
                    state=ocp.args.PyTreeRestore(),
                    loader=grain.PyGrainCheckpointRestore(loader_iter),
                    wirings_constants=ocp.args.PyTreeRestore(),
                ),
            ),
        )
        if latest_step is not None
        else None
    )
