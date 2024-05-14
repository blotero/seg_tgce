import tensorflow as tf
from keras.models import Model

from seg_tgce.data.utils import map_dataset_multiple_annotators

from .oxford_iiit_pet import OxfordIiitPet

MODEL_ORIGINAL_SHAPE = (256, 256)


def get_data_multiple_annotators(
    annotation_models: list[Model],
    target_shape: tuple[int, int] = (256, 256),
    batch_size: int = 32,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    dataset = OxfordIiitPet()
    train_dataset, val_dataset, test_dataset = dataset()
    train_data, val_data, test_data = (
        map_dataset_multiple_annotators(
            dataset=data,
            target_shape=target_shape,
            model_shape=MODEL_ORIGINAL_SHAPE,
            batch_size=batch_size,
            disturbance_models=annotation_models,
        )
        for data in (train_dataset, val_dataset, test_dataset)
    )
    return train_data, val_data, test_data
