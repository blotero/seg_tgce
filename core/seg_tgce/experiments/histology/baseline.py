import argparse

from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from seg_tgce.data.crowd_seg.tfds_builder import N_CLASSES, get_processed_data_baseline
from seg_tgce.experiments.plot_utils import plot_training_history, print_test_metrics
from seg_tgce.models.builders import (
    build_baseline_model_from_hparams,
)
from seg_tgce.models.ma_model import BaselineVisualizationCallback

from ..utils import handle_training

TARGET_SHAPE = (256, 256)
BATCH_SIZE = 32
TRAIN_EPOCHS = 20
TUNER_EPOCHS = 2
MAX_TRIALS = 5

DEFAULT_HPARAMS = {
    "q": 0.6,
    "noise_tolerance": 0.2,
    "dropout_rate": 0.2,
}


def build_model(hp=None):
    if hp is None:
        params = DEFAULT_HPARAMS
    else:
        params = {
            "q": hp.Float("q", min_value=0.1, max_value=0.9, step=0.1),
            "noise_tolerance": hp.Float(
                "noise_tolerance", min_value=0.1, max_value=0.9, step=0.1
            ),
            "dropout_rate": hp.Float(
                "dropout_rate", min_value=0.0, max_value=0.5, step=0.1
            ),
        }

    return build_baseline_model_from_hparams(
        learning_rate=1e-3,
        q=params["q"],
        noise_tolerance=params["noise_tolerance"],
        num_classes=N_CLASSES,
        target_shape=TARGET_SHAPE,
        dropout_rate=params["dropout_rate"],
    )


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

    processed_train, processed_validation, processed_test = get_processed_data_baseline(
        image_size=TARGET_SHAPE,
        batch_size=BATCH_SIZE,
        # use_augmentation=True,
        # augmentation_factor=2,
    )

    model = handle_training(
        processed_train,
        processed_validation,
        model_builder=build_model,
        use_tuner=args.use_tuner,
        tuner_epochs=TUNER_EPOCHS,
        objective="val_dice_coefficient",
        tuner_max_trials=MAX_TRIALS,
    )

    vis_callback = BaselineVisualizationCallback(processed_validation)

    lr_scheduler = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=4,
        min_lr=1e-6,
        mode="min",
        verbose=1,
    )

    print("\nTraining final model...")

    history = model.fit(
        processed_train,
        epochs=TRAIN_EPOCHS,
        validation_data=processed_validation,
        callbacks=[
            vis_callback,
            lr_scheduler,
            EarlyStopping(
                monitor="val_dice_coefficient",
                patience=3,
                mode="max",
                restore_best_weights=True,
            ),
        ],
    )

    plot_training_history(
        history,
        "Histology Baseline Model Training History",
        ["loss", "dice_coefficient", "jaccard_coefficient"],
    )
    print_test_metrics(
        model,
        processed_test,
        "Histology Baseline Model",
    )
