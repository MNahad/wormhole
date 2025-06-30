# Copyright 2024-2025 Mohammed Nawabuddin
# SPDX-License-Identifier: Apache-2.0

import os
from os import path
import unittest

import jax
import optax

from wormhole.config import config
from wormhole.training.evaluator import eval
from wormhole.training.loader import get_sample_dataloader
from wormhole.training.model_gen import get_models
from wormhole.training.trainer import create_train_state_and_constants, train
from . import synth


class ModelTrainingTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        synth.use_tmp_dir("train")

    @classmethod
    def tearDownClass(cls) -> None:
        synth.clear_tmp_dir()

    def setUp(self) -> None:
        training_conf = config()["training"]
        gold_dir = path.join(
            *(
                config()["data"]["catalogue"]["path"]
                + config()["data"]["catalogue"]["gold"]["path"]
            )
        )
        self.rngs = {"params": jax.random.key(0)}
        self.dataloader = get_sample_dataloader(
            gold_dir,
            gold_dir,
            batch_size=training_conf["batch_size"],
            allowed_labels_by_split=training_conf["allowed_labels_by_split"],
        ).get()[0]
        model = get_models()[training_conf["active_model"]]()
        tx = optax.adam(training_conf["hyperparameters"]["adam"])
        self.train_state, self.wirings_constants = (
            create_train_state_and_constants(
                model,
                self.rngs,
                tx,
                next(iter(self.dataloader)),
            )
        )
        self.checkpoint_dirname = path.join(
            *(config()["checkpoints"]["path"] + ("test",))
        )

    def test_a_exec_training(self) -> None:
        count = 0
        for (step, loss), _ in train(
            self.rngs,
            iter(self.dataloader),
            self.train_state,
            self.wirings_constants,
            "test",
        ):
            count += 1
            self.assertEqual(step, count)
            self.assertGreaterEqual(loss, 0)

    def test_b_exec_eval(self) -> None:
        checkpoints = os.listdir(self.checkpoint_dirname)
        for checkpoint_type in checkpoints:
            self.assertIn(checkpoint_type, ["checkpoint", "constants"])
        for (prediction, truth), confusion_matrix in eval(
            self.rngs,
            iter(self.dataloader),
            self.train_state,
            self.wirings_constants,
            "test",
        ):
            for value in prediction:
                self.assertIn(value, [True, False])
            for value in truth:
                self.assertIn(value, [True, False])
            for value in confusion_matrix:
                self.assertGreaterEqual(value, 0)


if __name__ == "__main__":
    unittest.main()
