from pathlib import Path
from typing import Callable

import keras_tuner as kt
import optuna
import tensorflow as tf
from keras import Model
from optuna.visualization import plot_optimization_history, plot_param_importances

from seg_tgce.experiments.types import HpTunerTrial


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
            train_gen,
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


def create_importance_visualizations(
    study: optuna.Study, save_dir: Path = Path("optuna_results")
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)

    try:
        fig = plot_param_importances(study)
        fig.write_image(save_dir / "parameter_importance.png")
        print(f"Parameter importance plot saved to {save_dir}/parameter_importance.png")
    except Exception as e:
        print(f"Could not create parameter importance plot: {e}")

    try:
        fig = plot_optimization_history(study)
        fig.write_image(save_dir / "optimization_history.png")
        print(f"Optimization history plot saved to {save_dir}/optimization_history.png")
    except Exception as e:
        print(f"Could not create optimization history plot: {e}")

    print("\n" + "=" * 50)
    print("HYPERPARAMETER IMPORTANCE RANKING")
    print("=" * 50)

    try:
        importances = optuna.importance.get_param_importances(study)
        for param, importance in sorted(
            importances.items(), key=lambda x: x[1], reverse=True
        ):
            print(f"{param}: {importance:.4f}")
    except Exception as e:
        print(f"Could not calculate parameter importances: {e}")


def handle_training_optuna(
    train: tf.data.Dataset,
    val: tf.data.Dataset,
    *,
    model_builder: Callable[[HpTunerTrial | None], Model],
    use_tuner: bool,
    tuner_epochs: int,
    objective: str,
    tuner_max_trials: int = 10,
    study_name: str = "experiment_hp_tuning",
) -> Model:
    print("Training with default hyperparameters...")

    def _objective(trial: optuna.Trial) -> float:
        print(f"\nTrial {trial.number}: Starting hyperparameter optimization...")

        model = model_builder(trial)

        history = model.fit(
            train,
            epochs=tuner_epochs,
            validation_data=val,
            verbose=0,
        )

        val_dice = max(history.history[objective])

        print(f"Trial {trial.number}: Validation Dice = {val_dice:.4f}")

        return val_dice

    if use_tuner:
        print("Using Keras Tuner for hyperparameter optimization...")
        study = optuna.study.create_study(
            study_name=study_name,
            direction="maximize",
        )

        print("Starting hyperparameter search...")
        study.optimize(_objective, n_trials=tuner_max_trials)

        best_hps = study.best_trial
        print("\nBest hyperparameters:")
        for param, value in best_hps.params.items():
            print(f"{param}: {value}")

        create_importance_visualizations(study)

        return model_builder(best_hps)

    print("Training with default hyperparameters...")
    return model_builder(None)
