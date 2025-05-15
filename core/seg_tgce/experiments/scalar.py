import keras_tuner as kt
from keras.optimizers import Adam

from seg_tgce.data.oxford_pet.oxford_pet import (
    fetch_models,
    get_data_multiple_annotators,
)
from seg_tgce.loss.tgce import TcgeScalar
from seg_tgce.metrics import DiceCoefficient, JaccardCoefficient
from seg_tgce.models.ma_model import VisualizationCallback
from seg_tgce.models.unet import unet_tgce_scalar

if __name__ == "__main__":
    TARGET_SHAPE = (128, 128)
    BATCH_SIZE = 8
    NUM_CLASSES = 3
    NOISE_LEVELS = [-20.0, 10.0]
    NUM_SCORERS = len(NOISE_LEVELS)

    learning_rate = 1e-3
    optimizer = Adam(learning_rate=learning_rate)

    loss_fn = TcgeScalar(
        num_classes=NUM_CLASSES,
        q=0.5,
        noise_tolerance=0.5,
        name="TGCE",
    )

    dice_fn = DiceCoefficient(num_classes=NUM_CLASSES)
    jaccard_fn = JaccardCoefficient(num_classes=NUM_CLASSES)

    model = unet_tgce_scalar(
        input_shape=TARGET_SHAPE + (3,),
        n_classes=NUM_CLASSES,
        n_scorers=NUM_SCORERS,
        name="Unet-TGCE-Scalar-Model",
    )

    model.compile(
        loss=loss_fn,
        metrics=[dice_fn],
        optimizer=optimizer,
    )
    model.loss_fn = loss_fn

    disturbance_models = fetch_models(NOISE_LEVELS)
    train, val, test = get_data_multiple_annotators(
        annotation_models=disturbance_models,
        target_shape=TARGET_SHAPE,
        batch_size=BATCH_SIZE,
        labeling_rate=1.0,
    )
    vis_callback = VisualizationCallback(val)

    history = model.fit(
        train.take(16).cache(),
        epochs=50,
        validation_data=val.take(8).cache(),
        callbacks=[vis_callback],
    )

    test_results = model.evaluate(test.cache())
    print(f"Test results: {test_results}")
