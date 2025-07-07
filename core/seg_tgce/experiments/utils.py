from typing import Callable

import keras_tuner as kt
import tensorflow as tf
from keras import Model


def handle_training(
    train: tf.data.Dataset,
    val: tf.data.Dataset,
    *,
    model_builder: Callable[[kt.HyperParameters | None], Model],
    use_tuner: bool,
    tuner_epochs: int,
    objective: str,
    tuner_max_trials: int = 10,
) -> Model:
    print("Training with default hyperparameters...")

    def train_directly() -> Model:
        return model_builder(None)

    def train_with_tuner(train_gen: tf.data.Dataset, val_gen: tf.data.Dataset) -> Model:
        tuner = kt.BayesianOptimization(
            model_builder,
            objective=kt.Objective(objective, direction="max"),
            max_trials=tuner_max_trials,
            directory="tuner_results",
            project_name="histology_scalar_tuning",
        )

        print("Starting hyperparameter search...")
        tuner.search(
            train_gen.take(10),
            epochs=tuner_epochs,
            validation_data=val_gen,
        )

        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        print("\nBest hyperparameters:")
        for param, value in best_hps.values.items():
            print(f"{param}: {value}")

        return model_builder(best_hps)

    if use_tuner:
        print("Using Keras Tuner for hyperparameter optimization...")
        model = train_with_tuner(train, val)
    else:
        print("Training with default hyperparameters...")
        model = train_directly()

    return model
