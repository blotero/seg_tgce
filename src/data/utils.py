import tensorflow as tf
from gcpds.image_segmentation.datasets.segmentation import OxfordIiitPet
from tensorflow.keras import Model  # pylint: disable=import-error,no-name-in-module


def disturb_mask(model: Model, image: tf.Tensor) -> tf.Tensor:
    return model(image)


def mix_channels(mask: tf.Tensor) -> tf.Tensor:
    return tf.stack([mask, 1 - mask], axis=-2)


def add_noisy_annotators(img: tf.Tensor, models: list[tf.Tensor]) -> tf.Tensor:
    return tf.transpose([disturb_mask(model, img) for model in models], [2, 3, 1, 4, 0])


def map_dataset_multiple_annotators(
    dataset: OxfordIiitPet,
    target_shape: tuple[int, int],
    batch_size: int,
    disturbance_models: list[Model],
) -> tf.Tensor:
    dataset_ = dataset.map(
        lambda img, mask, label, id_img: (img, mask),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    dataset_ = dataset_.map(
        lambda img, mask: (
            tf.image.resize(img, target_shape),
            tf.image.resize(mask, target_shape),
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    dataset_ = dataset_.map(
        lambda img, mask: (
            tf.image.resize(img, target_shape),
            tf.image.resize(mask, target_shape),
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    dataset_ = dataset_.map(
        lambda img, mask: (
            img,
            add_noisy_annotators(tf.expand_dims(img, 0), disturbance_models),
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    dataset_ = dataset_.map(
        lambda img, mask: (
            img,
            tf.reshape(mask, (mask.shape[0], mask.shape[1], 1, mask.shape[-1])),
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    dataset_ = dataset_.map(
        lambda img, mask: (img, mix_channels(mask)),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    dataset_ = dataset_.map(
        lambda img, mask: (img, tf.squeeze(mask, axis=2)),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    dataset_ = dataset_.batch(batch_size)
    return dataset_
