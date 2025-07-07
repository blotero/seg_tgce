from keras.models import load_model

from seg_tgce.data.oxford_pet.disturbance.model import (
    download_base_model,
    find_last_encoder_conv_layer,
    produce_disturbed_models,
)


def test_disturbance() -> None:
    snr_values = [10.0, 5.0, 2.0, 0.0, -5.0]
    model_path = download_base_model()
    model_ann = load_model(model_path, compile=False)

    encoder_layer_to_disturb = find_last_encoder_conv_layer(model_ann)

    disturbance_models, measured_snr_values = produce_disturbed_models(
        snr_values, model_path, encoder_layer_to_disturb
    )
    assert len(disturbance_models) == len(snr_values) == len(measured_snr_values)
