import argparse

import keras_tuner as kt
from keras import Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from seg_tgce.data.oxford_pet.oxford_pet import (
    fetch_models,
    get_data_multiple_annotators,
)
from seg_tgce.experiments.utils import handle_training
from seg_tgce.experiments.plot_utils import plot_training_history, print_test_metrics
from seg_tgce.models.builders import build_scalar_model_from_hparams
from seg_tgce.models.ma_model import ScalarVisualizationCallback

TARGET_SHAPE = (256, 256)
BATCH_SIZE = 16
NUM_CLASSES = 3
NOISE_LEVELS = [-20.0, 10.0]
NUM_SCORERS = len(NOISE_LEVELS)
TRAIN_EPOCHS = 50
TUNER_EPOCHS = 10
SEED = 42
LABELING_RATE = 0.5
TUNER_MAX_TRIALS = 10

DEFAULT_HPARAMS = {
    "initial_learning_rate": 1e-3,
    "q": 0.7,
    "noise_tolerance": 0.5,
    "a": 0.2,
    "b": 0.7,
}


def build_model(hp: kt.HyperParameters | None = None) -> Model:
    """Build model using hyperparameters.

    Args:
        hp: Optional Keras Tuner hyperparameters object. If None, uses default values.

    Returns:
        Compiled Keras model
    """
    if hp is None:
        params = DEFAULT_HPARAMS
    else:
        params = {
            "initial_learning_rate": DEFAULT_HPARAMS["initial_learning_rate"],
            "q": hp.Float("q", min_value=0.1, max_value=0.9, step=0.1),
            "noise_tolerance": hp.Float(
                "noise_tolerance", min_value=0.1, max_value=0.9, step=0.1
            ),
            "b": hp.Float("b", min_value=0.1, max_value=1.0, step=0.1),
            "a": hp.Float("a", min_value=0.1, max_value=1.0, step=0.1),
        }

    return build_scalar_model_from_hparams(
        learning_rate=params["initial_learning_rate"],
        q=params["q"],
        noise_tolerance=params["noise_tolerance"],
        b=params["b"],
        a=params["a"],
        num_classes=NUM_CLASSES,
        target_shape=TARGET_SHAPE,
        n_scorers=NUM_SCORERS,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train pets scalar model with or without hyperparameter tuning"
    )
    parser.add_argument(
        "--use-tuner",
        action="store_true",
        help="Use Keras Tuner for hyperparameter optimization",
    )
    args = parser.parse_args()

    disturbance_models = fetch_models(NOISE_LEVELS, seed=SEED)
    train, val, test = get_data_multiple_annotators(
        annotation_models=disturbance_models,
        target_shape=TARGET_SHAPE,
        batch_size=BATCH_SIZE,
        labeling_rate=LABELING_RATE,
    )

    model = handle_training(
        train,
        val,
        model_builder=build_model,
        use_tuner=args.use_tuner,
        tuner_epochs=TUNER_EPOCHS,
        objective="val_segmentation_output_dice_coefficient",
        tuner_max_trials=TUNER_MAX_TRIALS,
    )

    vis_callback = ScalarVisualizationCallback(val, save_dir="vis/pets/scalar")

    lr_scheduler = ReduceLROnPlateau(
        monitor="val_segmentation_output_dice_coefficient",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        mode="max",
        verbose=1,
    )

    print("\nTraining final model...")
    history = model.fit(
        train,
        epochs=TRAIN_EPOCHS,
        validation_data=val.cache(),
        callbacks=[
            vis_callback,
            lr_scheduler,
            EarlyStopping(
                monitor="val_segmentation_output_dice_coefficient",
                patience=5,
                mode="max",
                restore_best_weights=True,
            ),
        ],
    )

    plot_training_history(history, "Pets Scalar Model Training History")
    print_test_metrics(model, test, "Pets Scalar")
