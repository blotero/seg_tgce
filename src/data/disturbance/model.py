from enum import Enum
import numpy as np
from tensorflow.keras import Model
import tensorflow as tf
import gdown
import os


def compute_snr(signal: float, noise_std: float) -> float:
    return float(10 * np.log10(np.mean(signal**2) / noise_std**2))


class SnrType(Enum):
    log = 0
    linear = 1


def add_noise_to_layer_weights(
    model: Model,
    layer_num: int,
    noise_snr: float,
    snr_type: SnrType = SnrType.log,
    verbose: int = 0,
) -> float:
    layer_weights = model.layers[layer_num].get_weights()

    sig_power = np.mean(layer_weights[0] ** 2)

    if snr_type == SnrType.log:
        noise_power = sig_power / (10 ** (noise_snr / 10))
    elif snr_type == SnrType.linear:
        noise_power = sig_power / noise_snr

    noise_std = noise_power ** (1 / 2)

    snr = compute_snr(layer_weights[0], noise_std)

    if verbose > 0:
        print(f"Adding noise for snr: {noise_snr}\n\n")
        print(f"Signal power: {sig_power}")
        print(f"Noise power: {noise_power}\n\n")

    for i in range(layer_weights[0].shape[0]):
        for j in range(layer_weights[0].shape[1]):
            layer_weights[0][i][j] += np.random.randn(128, 128) * noise_std

    model.layers[layer_num].set_weights(layer_weights)
    return snr


def produce_disturbed_models(
    snr_values: list[int], base_model_path: str, last_conv_encoder_layer: int
) -> tuple[list[Model], list[float]]:
    snr_measured_values: list[float] = []
    models: list[Model] = []

    for value in snr_values:
        model_: Model = tf.keras.models.load_model(base_model_path, compile=False)
        snr = add_noise_to_layer_weights(model_, last_conv_encoder_layer, value)
        snr_measured_values.append(snr)
        models.append(model_)
    return models, snr_measured_values


def download_base_model() -> str:
    model_url = "https://drive.google.com/uc?id=1x39L3QNDMye1SJhKh1gf4YS-HRFLTs6G"
    gdown.download(model_url)
    model_extension = "keras"
    paths = []

    for file in os.listdir("."):
        if file.endswith(model_extension):
            paths.append(file)

    model_path = paths[0]
    return model_path


def find_last_encoder_conv_layer(model: Model) -> tf.keras.layers:
    last_conv_encoder_layer = 0
    for i, layer in enumerate(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_encoder_layer = i
        if isinstance(layer, tf.keras.layers.UpSampling2D):
            break
    return last_conv_encoder_layer


if __name__ == "__main__":
    snr_values = [10, 5, 2, 0, -5]
    model_path = download_base_model()
    model_ann = tf.keras.models.load_model(model_path, compile=False)

    last_conv_encoder_layer = find_last_encoder_conv_layer(model_ann)

    disturbance_models, measured_snr_values = produce_disturbed_models(
        snr_values, model_path, last_conv_encoder_layer
    )
    print(f"Measured snr values: {measured_snr_values}")
