import tensorflow as tf

from data.disturbance.model import (
    download_base_model,
    find_last_encoder_conv_layer,
    produce_disturbed_models,
)
from data.oxford_pet.oxford_pet import get_data_multiple_annotators


def test_oxford_pet_flow() -> None:
    snr_values = [10, 5, 2, 0, -5]
    model_path = download_base_model()
    model_ann = tf.keras.models.load_model(  # pylint: disable=no-member
        model_path, compile=False
    )

    last_conv_encoder_layer = find_last_encoder_conv_layer(model_ann)

    disturbance_models, measured_snr_values = produce_disturbed_models(
        snr_values, model_path, last_conv_encoder_layer
    )
    print(f"Measured snr values: {measured_snr_values}")
    train, val, test = get_data_multiple_annotators(disturbance_models)
    for data in [train, val, test]:
        assert data
