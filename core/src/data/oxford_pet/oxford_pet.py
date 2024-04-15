import tensorflow as tf
from gcpds.image_segmentation.datasets.segmentation import OxfordIiitPet
from keras.models import Model

from data.utils import map_dataset_multiple_annotators


def get_data_multiple_annotators(
    annotation_models: list[Model],
    target_shape: tuple[int, int] = (256, 256),
    batch_size: int = 32,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    dataset = OxfordIiitPet()
    train_dataset, val_dataset, test_dataset = dataset()
    train_data = map_dataset_multiple_annotators(
        train_dataset, target_shape, batch_size, annotation_models
    )
    val_data = map_dataset_multiple_annotators(
        val_dataset, target_shape, batch_size, annotation_models
    )
    test_data = map_dataset_multiple_annotators(
        test_dataset, target_shape, batch_size, annotation_models
    )
    return train_data, val_data, test_data
