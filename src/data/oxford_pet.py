import tensorflow as tf
from disturbance.model import (
    download_base_model,
    find_last_encoder_conv_layer,
    produce_disturbed_models,
)
from gcpds.image_segmentation.datasets.segmentation import OxfordIiitPet
from tensorflow.keras import Model
from utils import map_dataset_MA


def get_data_MA(
    disturbance_models: list[Model],
    target_shape: tuple[int, int] = (256, 256),
    batch_size: int = 32,
    num_annotators: int = 5,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    dataset = OxfordIiitPet()
    train_dataset, val_dataset, test_dataset = dataset()
    train = map_dataset_MA(
        train_dataset, target_shape, batch_size, num_annotators, disturbance_models
    )
    val = map_dataset_MA(
        val_dataset, target_shape, batch_size, num_annotators, disturbance_models
    )
    test = map_dataset_MA(
        test_dataset, target_shape, batch_size, num_annotators, disturbance_models
    )
    return train, val, test


if __name__ == "__main__":
    snr_values = [10, 5, 2, 0, -5]
    model_path = download_base_model()
    model_ann = tf.keras.models.load_model(model_path, compile=False)

    last_conv_encoder_layer = find_last_encoder_conv_layer(model_ann)

    disturbance_models, measured_snr_values = produce_disturbed_models(
        snr_values, model_path, last_conv_encoder_layer
    )
    print(f"Measured snr values: {measured_snr_values}")
    train, val, test = get_data_MA(disturbance_models)
    print(f"Train data: {train}")
    print(f"Val data: {val}")
    print(f"Test data: {test}")
