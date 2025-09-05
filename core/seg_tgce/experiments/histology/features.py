import argparse

from keras import Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from seg_tgce.data.crowd_seg.tfds_builder import (
    N_CLASSES,
    N_REAL_SCORERS,
    get_processed_data,
)
from seg_tgce.experiments.plot_utils import plot_training_history, print_test_metrics
from seg_tgce.experiments.types import HpTunerTrial
from seg_tgce.models.builders import build_features_model_from_hparams
from seg_tgce.models.ma_model import FeatureVisualizationCallback

from ..utils import handle_training_optuna

TARGET_SHAPE = (256, 256)
BATCH_SIZE = 4
TRAIN_EPOCHS = 20
TUNER_EPOCHS = 1
TUNER_MAX_TRIALS = 10
STUDY_NAME = "histology_features_tuning"
OBJECTIVE = "val_segmentation_output_dice_coefficient"

DEFAULT_HPARAMS = {
    "initial_learning_rate": 1e-3,
    "q": 0.5,
    "noise_tolerance": 0.5,
    "a": 0.5,
    "b": 0.5,
    "lambda_reg_weight": 0.1,
    "lambda_entropy_weight": 0.1,
    "lambda_sum_weight": 0.1,
}


def build_model_from_trial(trial: HpTunerTrial | None) -> Model:
    if trial is None:
        return build_features_model_from_hparams(
            learning_rate=DEFAULT_HPARAMS["initial_learning_rate"],
            q=DEFAULT_HPARAMS["q"],
            noise_tolerance=DEFAULT_HPARAMS["noise_tolerance"],
            b=DEFAULT_HPARAMS["b"],
            a=DEFAULT_HPARAMS["a"],
            lambda_reg_weight=DEFAULT_HPARAMS["lambda_reg_weight"],
            lambda_entropy_weight=DEFAULT_HPARAMS["lambda_entropy_weight"],
            lambda_sum_weight=DEFAULT_HPARAMS["lambda_sum_weight"],
            num_classes=N_CLASSES,
            target_shape=TARGET_SHAPE,
            n_scorers=N_REAL_SCORERS,
        )

    return build_features_model_from_hparams(
        learning_rate=trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
        q=trial.suggest_float("q", 0.1, 0.9, step=0.01),
        noise_tolerance=trial.suggest_float("noise_tolerance", 0.1, 0.9, step=0.01),
        a=trial.suggest_float("a", 0.1, 10.0, step=0.1),
        b=trial.suggest_float("b", 0.1, 0.99, step=0.01),
        lambda_reg_weight=trial.suggest_float("lambda_reg_weight", 0.0, 10.0, step=0.1),
        lambda_entropy_weight=trial.suggest_float(
            "lambda_entropy_weight", 0.0, 10.0, step=0.1
        ),
        lambda_sum_weight=trial.suggest_float("lambda_sum_weight", 0.0, 10.0, step=0.1),
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
        image_size=TARGET_SHAPE, batch_size=BATCH_SIZE, use_augmentation=False
    )

    model, study = handle_training_optuna(
        processed_train,
        processed_validation,
        model_builder=build_model_from_trial,
        use_tuner=args.use_tuner,
        tuner_epochs=TUNER_EPOCHS,
        objective=OBJECTIVE,
    )

    vis_callback = FeatureVisualizationCallback(
        processed_validation, save_dir="vis/histology/features"
    )

    lr_scheduler = ReduceLROnPlateau(
        monitor=OBJECTIVE,
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
                monitor=OBJECTIVE,
                patience=5,
                mode="max",
                restore_best_weights=True,
            ),
        ],
    )

    plot_training_history(history, "Histology Features Model Training History")
    print_test_metrics(model, processed_test, "Histology Features")
