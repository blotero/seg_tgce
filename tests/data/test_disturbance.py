import tensorflow as tf

from data.disturbance.model import (
    download_base_model,
    find_last_encoder_conv_layer,
    produce_disturbed_models,
)


def test_disturbance() -> None:
    snr_values = [10, 5, 2, 0, -5]
    model_path = download_base_model()
    model_ann = tf.keras.models.load_model(  # pylint: disable=no-member
        model_path, compile=False
    )

    encoder_layer_to_disturb = find_last_encoder_conv_layer(model_ann)

    disturbance_models, measured_snr_values = produce_disturbed_models(
        snr_values, model_path, encoder_layer_to_disturb
    )
    assert len(disturbance_models) == len(snr_values) == len(measured_snr_values)
