from keras import Model
from keras.optimizers import Adam

from seg_tgce.loss.tgce import TgceBaseline, TgceFeatures, TgcePixel, TgceScalar
from seg_tgce.metrics import DiceCoefficient, JaccardCoefficient
from seg_tgce.models.unet import (
    unet_baseline,
    unet_tgce_features,
    unet_tgce_pixel,
    unet_tgce_scalar,
)


def build_baseline_model_from_hparams(
    learning_rate: float,
    q: float,
    noise_tolerance: float,
    num_classes: int,
    target_shape: tuple,
    dropout_rate: float = 0.2,
) -> Model:
    """Build the baseline model with direct hyperparameter values."""

    optimizer = Adam(learning_rate=learning_rate)

    loss_fn = TgceBaseline(
        num_classes=num_classes,
        q=q,
        noise_tolerance=noise_tolerance,
        name="TGCE",
    )

    dice_fn = DiceCoefficient(
        num_classes=num_classes,
        name="dice_coefficient",
    )
    jaccard_fn = JaccardCoefficient(
        num_classes=num_classes,
        name="jaccard_coefficient",
    )

    model = unet_baseline(
        input_shape=target_shape + (3,),
        n_classes=num_classes,
        name="Unet-Baseline-Model",
        dropout_rate=dropout_rate,
    )

    model.compile(
        loss=loss_fn,
        metrics=[dice_fn, jaccard_fn],
        optimizer=optimizer,
    )
    return model


def build_scalar_model_from_hparams(
    *,
    learning_rate: float,
    q: float,
    noise_tolerance: float,
    a: float,
    b: float,
    c: float,
    lambda_reg_weight: float,
    lambda_entropy_weight: float,
    lambda_sum_weight: float,
    num_classes: int,
    target_shape: tuple,
    n_scorers: int,
) -> Model:
    optimizer = Adam(learning_rate=learning_rate)

    loss_fn = TgceScalar(
        num_classes=num_classes,
        q=q,
        noise_tolerance=noise_tolerance,
        a=a,
        b=b,
        c=c,
        lambda_reg_weight=lambda_reg_weight,
        lambda_entropy_weight=lambda_entropy_weight,
        lambda_sum_weight=lambda_sum_weight,
        name="TGCE",
    )

    dice_fn = DiceCoefficient(
        num_classes=num_classes,
        name="dice_coefficient",
    )
    jaccard_fn = JaccardCoefficient(
        num_classes=num_classes,
        name="jaccard_coefficient",
    )

    model = unet_tgce_scalar(
        input_shape=target_shape + (3,),
        n_classes=num_classes,
        n_scorers=n_scorers,
        name="Unet-TGCE-Scalar-Model",
    )

    model.compile(
        loss=loss_fn,
        metrics={"segmentation_output": [dice_fn, jaccard_fn]},
        optimizer=optimizer,
    )
    model.loss_fn = loss_fn
    return model


def build_features_model_from_hparams(
    learning_rate: float,
    q: float,
    noise_tolerance: float,
    lambda_reg_weight: float,
    lambda_entropy_weight: float,
    lambda_sum_weight: float,
    num_classes: int,
    target_shape: tuple,
    n_scorers: int,
) -> Model:
    """Build the features model with direct hyperparameter values.

    Args:
        learning_rate: Learning rate for the optimizer
        q: q parameter for TGCE loss
        noise_tolerance: Noise tolerance parameter for TGCE loss
        lambda_reg_weight: Regularization weight for TGCE loss
        lambda_entropy_weight: Entropy weight for TGCE loss
        lambda_sum_weight: Sum weight for TGCE loss
        num_classes: Number of classes in the segmentation
        target_shape: Target shape of input images
        n_scorers: Number of annotators/scorers

    Returns:
        Compiled Keras model
    """
    optimizer = Adam(learning_rate=learning_rate)

    loss_fn = TgceFeatures(
        num_classes=num_classes,
        q=q,
        noise_tolerance=noise_tolerance,
        lambda_reg_weight=lambda_reg_weight,
        lambda_entropy_weight=lambda_entropy_weight,
        lambda_sum_weight=lambda_sum_weight,
        name="TGCE_FEATURES",
    )

    dice_fn = DiceCoefficient(
        num_classes=num_classes,
        name="dice_coefficient",
    )
    jaccard_fn = JaccardCoefficient(
        num_classes=num_classes,
        name="jaccard_coefficient",
    )

    model = unet_tgce_features(
        input_shape=target_shape + (3,),
        n_classes=num_classes,
        n_scorers=n_scorers,
        name="Unet-TGCE-Features-Model",
    )

    model.compile(
        loss=loss_fn,
        metrics={"segmentation_output": [dice_fn, jaccard_fn]},
        optimizer=optimizer,
    )
    model.loss_fn = loss_fn
    return model


def build_pixel_model_from_hparams(
    learning_rate: float,
    q: float,
    noise_tolerance: float,
    lambda_reg_weight: float,
    lambda_entropy_weight: float,
    lambda_sum_weight: float,
    num_classes: int,
    target_shape: tuple,
    n_scorers: int,
) -> Model:
    optimizer = Adam(learning_rate=learning_rate)

    loss_fn = TgcePixel(
        num_classes=num_classes,
        q=q,
        noise_tolerance=noise_tolerance,
        lambda_reg_weight=lambda_reg_weight,
        lambda_entropy_weight=lambda_entropy_weight,
        lambda_sum_weight=lambda_sum_weight,
        name="TGCE_PIXEL",
    )

    dice_fn = DiceCoefficient(
        num_classes=num_classes,
        name="dice_coefficient",
    )
    jaccard_fn = JaccardCoefficient(
        num_classes=num_classes,
        name="jaccard_coefficient",
    )

    model = unet_tgce_pixel(
        input_shape=target_shape + (3,),
        n_classes=num_classes,
        n_scorers=n_scorers,
        name="Unet-TGCE-Pixel-Model",
    )

    model.compile(
        loss=loss_fn,
        metrics={"segmentation_output": [dice_fn, jaccard_fn]},
        optimizer=optimizer,
    )
    model.loss_fn = loss_fn
    return model
