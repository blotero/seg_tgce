import argparse

import keras_tuner as kt
from seg_tgce.data.crowd_seg.generator import (
    REAL_SCORERS,
    get_crowd_seg_data,
)
from seg_tgce.experiments.plot_utils import plot_training_history, print_test_metrics
from seg_tgce.models.builders import build_scalar_model_from_hparams
from seg_tgce.models.ma_model import ScalarVisualizationCallback

TARGET_SHAPE = (64, 64)
BATCH_SIZE = 128
NUM_CLASSES = 6  # From CLASSES_DEFINITION in generator.py
TRAIN_EPOCHS = 5
TUNER_EPOCHS = 1
N_SCORERS = len(REAL_SCORERS)

# Default hyperparameters for direct training
DEFAULT_HPARAMS = {
    "learning_rate": 1e-3,
    "q": 0.5,
    "noise_tolerance": 0.5,
    "lambda_reg_weight": 0.1,
    "lambda_entropy_weight": 0.1,
    "lambda_sum_weight": 0.1,
}


def build_model(hp=None):
    """Build model using hyperparameters.

    Args:
        hp: Optional Keras Tuner hyperparameters object. If None, uses default values.

    Returns:
        Compiled Keras model
    """
    if hp is None:
        # Use default hyperparameters
        params = DEFAULT_HPARAMS
    else:
        # Use tuner hyperparameters
        params = {
            "learning_rate": hp.Float(
                "learning_rate", min_value=1e-5, max_value=1e-2, sampling="LOG"
            ),
            "q": hp.Float("q", min_value=0.1, max_value=0.9, step=0.1),
            "noise_tolerance": hp.Float(
                "noise_tolerance", min_value=0.1, max_value=0.9, step=0.1
            ),
            "lambda_reg_weight": hp.Float(
                "lambda_reg_weight", min_value=0.01, max_value=0.2, step=0.01
            ),
            "lambda_entropy_weight": hp.Float(
                "lambda_entropy_weight", min_value=0.01, max_value=0.2, step=0.01
            ),
            "lambda_sum_weight": hp.Float(
                "lambda_sum_weight", min_value=0.01, max_value=0.2, step=0.01
            ),
        }

    return build_scalar_model_from_hparams(
        learning_rate=params["learning_rate"],
        q=params["q"],
        noise_tolerance=params["noise_tolerance"],
        lambda_reg_weight=params["lambda_reg_weight"],
        lambda_entropy_weight=params["lambda_entropy_weight"],
        lambda_sum_weight=params["lambda_sum_weight"],
        num_classes=NUM_CLASSES,
        target_shape=TARGET_SHAPE,
        n_scorers=N_SCORERS,
    )


def train_with_tuner(train_gen, val_gen):
    """Train model using Keras Tuner for hyperparameter optimization."""
    tuner = kt.BayesianOptimization(
        build_model,
        objective=kt.Objective(
            "val_segmentation_output_dice_coefficient", direction="max"
        ),
        max_trials=10,
        directory="tuner_results",
        project_name="histology_scalar_tuning",
    )

    print("Starting hyperparameter search...")
    tuner.search(
        train_gen.take(10),
        epochs=TUNER_EPOCHS,
        validation_data=val_gen,
    )

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("\nBest hyperparameters:")
    for param, value in best_hps.values.items():
        print(f"{param}: {value}")

    return build_model(best_hps)


def train_directly():
    """Train model using default hyperparameters."""
    return build_model()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train histology scalar model with or without hyperparameter tuning"
    )
    parser.add_argument(
        "--use-tuner",
        action="store_true",
        help="Use Keras Tuner for hyperparameter optimization",
    )
    args = parser.parse_args()

    train_gen, val_gen, test_gen = get_crowd_seg_data(
        image_size=TARGET_SHAPE,
        batch_size=BATCH_SIZE,
    )

    if args.use_tuner:
        print("Using Keras Tuner for hyperparameter optimization...")
        model = train_with_tuner(train_gen, val_gen)
    else:
        print("Training with default hyperparameters...")
        model = train_directly()

    vis_callback = ScalarVisualizationCallback(val_gen)

    print("\nTraining final model...")

    history = model.fit(
        train_gen,
        epochs=TRAIN_EPOCHS,
        validation_data=val_gen,
        callbacks=[vis_callback],
    )

    plot_training_history(history, "Histology Scalar Model Training History")
    print_test_metrics(model, test_gen, "Histology Scalar")
