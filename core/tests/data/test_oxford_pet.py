from keras.models import load_model

from core.seg_tgce.data.oxford_pet.disturbance.model import (
    download_base_model,
    find_last_encoder_conv_layer,
    produce_disturbed_models,
)
from core.seg_tgce.data.oxford_pet.oxford_pet import get_data_multiple_annotators


def test_oxford_pet_flow() -> None:
    snr_values = [10.0, 5.0, 2.0, 0.0, -5.0]
    model_path = download_base_model()
    model_ann = load_model(model_path, compile=False)

    last_conv_encoder_layer = find_last_encoder_conv_layer(model_ann)

    disturbance_models, measured_snr_values = produce_disturbed_models(
        snr_values, model_path, last_conv_encoder_layer
    )
    print(f"Measured snr values: {measured_snr_values}")
    train, val, test = get_data_multiple_annotators(disturbance_models)
    for data in [train, val, test]:
        assert data
