# Copyright 2024-2025 Mohammed Nawabuddin
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timezone
from functools import wraps
from os import path
from typing import Callable, Iterator, Mapping, Optional

import flax.core
from flax.training import train_state
import grain.python as grain
import jax
from jax import Array
import orbax.checkpoint as ocp

from wormhole.dataset import LightCurve

type _ReturnType = Iterator[
    tuple[
        tuple[int, Array],
        tuple[Iterator[LightCurve], train_state.TrainState],
    ]
]

type _OuterFn = Callable[
    [
        Iterator[LightCurve],
        dict[str, Array],
        train_state.TrainState,
        flax.core.FrozenDict,
        Optional[str],
    ],
    _ReturnType,
]

type _InnerFn = Callable[
    [
        Iterator[LightCurve],
        dict[str, Array],
        train_state.TrainState,
        flax.core.FrozenDict,
        int,
    ],
    _ReturnType,
]


def enable_checkpointing(
    dir: str,
    options: Optional[ocp.CheckpointManagerOptions] = None,
    default_job_id: str = datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    ),
) -> Callable[[_InnerFn], _OuterFn]:

    def checkpointer(fn: _InnerFn) -> _OuterFn:

        @wraps(fn)
        def wrapper(dataset, rngs, state, constants, job_id):
            ckpt_path = path.abspath(
                path.join(dir, job_id if job_id else default_job_id)
            )
            with ocp.CheckpointManager(
                path.join(ckpt_path, "constants"),
                options=options if options else ocp.CheckpointManagerOptions(),
            ) as constants_mngr:
                restored_constants = _restore(
                    constants_mngr,
                    ocp.args.StandardRestore(_pytree_to_abstract(constants)),
                )
                if restored_constants is None:
                    _save(constants_mngr, 0, ocp.args.StandardSave(constants))
                else:
                    constants = restored_constants[1]
            with ocp.CheckpointManager(
                path.join(ckpt_path, "checkpoint"),
                options=options if options else ocp.CheckpointManagerOptions(),
            ) as mngr:
                step = 0
                restored = _restore(
                    mngr,
                    ocp.args.Composite(
                        state=ocp.args.StandardRestore(
                            _pytree_to_abstract(state)
                        ),
                        loader=grain.PyGrainCheckpointRestore(dataset),
                    ),
                )
                if restored is not None:
                    step = restored[0]
                    restored_composite = restored[1]
                    dataset = restored_composite["loader"]
                    state = restored_composite["state"]
                for next_iter in fn(dataset, rngs, state, constants, step):
                    mngr.wait_until_finished()
                    _save(
                        mngr,
                        next_iter[0][0],
                        ocp.args.Composite(
                            state=ocp.args.StandardSave(next_iter[1][1]),
                            loader=grain.PyGrainCheckpointSave(
                                next_iter[1][0]
                            ),
                        ),
                    )
                    yield next_iter

        return wrapper

    return checkpointer


def _save(
    mngr: ocp.CheckpointManager,
    step: int,
    save_args: ocp.args.CheckpointArgs,
) -> None:
    mngr.save(step, args=save_args)


def _restore(
    mngr: ocp.CheckpointManager,
    restore_args=ocp.args.CheckpointArgs,
) -> Optional[tuple[int, Mapping]]:
    latest_step = mngr.latest_step()
    return (
        (
            latest_step,
            mngr.restore(latest_step, args=restore_args),
        )
        if latest_step is not None
        else None
    )


def _pytree_to_abstract(tree: Array) -> jax.ShapeDtypeStruct:
    return jax.tree.map(ocp.tree.to_shape_dtype_struct, tree)
