import keras_tuner as kt
import tensorflow as tf

from seg_tgce.data.oxford_pet.oxford_pet import (
    fetch_models,
    get_data_multiple_annotators,
)
from seg_tgce.experiments.plot_utils import plot_training_history, print_test_metrics
from seg_tgce.models.builders import build_pixel_model_from_hparams
from seg_tgce.models.ma_model import PixelVisualizationCallback

TARGET_SHAPE = (256, 256)
BATCH_SIZE = 16
NUM_CLASSES = 3
NOISE_LEVELS = [-20.0, 10.0]
NUM_SCORERS = len(NOISE_LEVELS)
TRAIN_EPOCHS = 50
TUNER_EPOCHS = 1
TUNER_TRIALS = 1


def build_model(hp: kt.HyperParameters) -> tf.keras.Model:
    learning_rate = hp.Float(
        "learning_rate", min_value=1e-5, max_value=1e-2, sampling="LOG"
    )
    q = hp.Float("q", min_value=0.1, max_value=0.9, step=0.1)
    noise_tolerance = hp.Float("noise_tolerance", min_value=0.1, max_value=0.9, step=0.1)
    lambda_reg_weight = hp.Float(
        "lambda_reg_weight", min_value=0.01, max_value=0.5, step=0.01
    )
    lambda_entropy_weight = hp.Float(
        "lambda_entropy_weight", min_value=0.01, max_value=0.5, step=0.01
    )
    lambda_sum_weight = hp.Float(
        "lambda_sum_weight", min_value=0.01, max_value=0.5, step=0.01
    )

    return build_pixel_model_from_hparams(
        learning_rate=learning_rate,
        q=q,
        noise_tolerance=noise_tolerance,
        lambda_reg_weight=lambda_reg_weight,
        lambda_entropy_weight=lambda_entropy_weight,
        lambda_sum_weight=lambda_sum_weight,
        num_classes=NUM_CLASSES,
        target_shape=TARGET_SHAPE,
        n_scorers=NUM_SCORERS,
    )


if __name__ == "__main__":
    disturbance_models = fetch_models(NOISE_LEVELS)
    train, val, test = get_data_multiple_annotators(
        annotation_models=disturbance_models,
        target_shape=TARGET_SHAPE,
        batch_size=BATCH_SIZE,
        labeling_rate=0.5,
    )

    tuner = kt.BayesianOptimization(
        build_model,
        objective=kt.Objective(
            "val_segmentation_output_dice_coefficient", direction="max"
        ),
        max_trials=TUNER_TRIALS,
        directory="tuner_results",
        project_name="pixel_tuning",
    )

    print("Starting hyperparameter search...")
    tuner.search(
        train.take(16).cache(),
        epochs=TUNER_EPOCHS,
        validation_data=val.take(8).cache(),
    )

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("\nBest hyperparameters:")
    for param, value in best_hps.values.items():
        print(f"{param}: {value}")

    model = build_model(best_hps)
    vis_callback = PixelVisualizationCallback(val)

    print("\nTraining with best hyperparameters...")
    history = model.fit(
        train.take(16).cache(),
        epochs=TRAIN_EPOCHS,
        validation_data=val.take(8).cache(),
        callbacks=[vis_callback],
    )

    plot_training_history(history, "Pixel Model Training History")

    print_test_metrics(model, test, "Pixel")
