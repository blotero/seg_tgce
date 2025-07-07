import argparse

from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from seg_tgce.data.crowd_seg.tfds_builder import (
    N_CLASSES,
    N_REAL_SCORERS,
    get_processed_data,
)
from seg_tgce.experiments.plot_utils import plot_training_history, print_test_metrics
from seg_tgce.models.builders import build_features_model_from_hparams
from seg_tgce.models.ma_model import FeatureVisualizationCallback

from ..utils import handle_training

TARGET_SHAPE = (256, 256)
BATCH_SIZE = 32
TRAIN_EPOCHS = 20
TUNER_EPOCHS = 1

DEFAULT_HPARAMS = {
    "initial_learning_rate": 1e-3,
    "q": 0.5,
    "noise_tolerance": 0.5,
    "lambda_reg_weight": 0.1,
    "lambda_entropy_weight": 0.1,
    "lambda_sum_weight": 0.1,
}


def build_model(hp=None):
    if hp is None:
        params = DEFAULT_HPARAMS
    else:
        params = {
            "initial_learning_rate": hp.Float(
                "learning_rate", min_value=1e-5, max_value=1e-2, sampling="LOG"
            ),
            "q": hp.Float("q", min_value=0.1, max_value=0.9, step=0.1),
            "noise_tolerance": hp.Float(
                "noise_tolerance", min_value=0.1, max_value=0.9, step=0.1
            ),
            "lambda_reg_weight": hp.Float(
                "lambda_reg_weight", min_value=0.01, max_value=0.5, step=0.01
            ),
            "lambda_entropy_weight": hp.Float(
                "lambda_entropy_weight", min_value=0.01, max_value=0.5, step=0.01
            ),
            "lambda_sum_weight": hp.Float(
                "lambda_sum_weight", min_value=0.01, max_value=0.5, step=0.01
            ),
        }

    return build_features_model_from_hparams(
        learning_rate=params["initial_learning_rate"],
        q=params["q"],
        noise_tolerance=params["noise_tolerance"],
        lambda_reg_weight=params["lambda_reg_weight"],
        lambda_entropy_weight=params["lambda_entropy_weight"],
        lambda_sum_weight=params["lambda_sum_weight"],
        num_classes=N_CLASSES,
        target_shape=TARGET_SHAPE,
        n_scorers=N_REAL_SCORERS,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train histology features model with or without hyperparameter tuning"
    )
    parser.add_argument(
        "--use-tuner",
        action="store_true",
        help="Use Keras Tuner for hyperparameter optimization",
    )
    args = parser.parse_args()

    processed_train, processed_validation, processed_test = get_processed_data(
        image_size=TARGET_SHAPE,
        batch_size=BATCH_SIZE,
        use_augmentation=True,
        augmentation_factor=2,
    )

    model = handle_training(
        processed_train,
        processed_validation,
        model_builder=build_model,
        use_tuner=args.use_tuner,
        tuner_epochs=TUNER_EPOCHS,
        objective="val_segmentation_output_dice_coefficient",
    )

    vis_callback = FeatureVisualizationCallback(
        processed_validation, save_dir="vis/histology/features"
    )

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
        processed_train,
        epochs=TRAIN_EPOCHS,
        validation_data=processed_validation,
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

    plot_training_history(history, "Histology Features Model Training History")
    print_test_metrics(model, processed_test, "Histology Features")
