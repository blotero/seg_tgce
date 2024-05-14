from typing import Tuple

import tensorflow as tf
from keras.models import Model
from tensorflow import Tensor


def disturb_mask(
    model: Model,
    image: Tensor,
    model_shape: Tuple[int, int],
    target_shape: Tuple[int, int],
) -> Tensor:
    return tf.image.resize(model(tf.image.resize(image, model_shape)), target_shape)


def mix_channels(mask: Tensor) -> Tensor:
    return tf.stack([mask, 1 - mask], axis=-2)


def add_noisy_annotators(
    img: Tensor,
    models: list[Tensor],
    model_shape: Tuple[int, int],
    target_shape: Tuple[int, int],
) -> Tensor:
    return tf.transpose(
        [
            disturb_mask(model, img, model_shape=model_shape, target_shape=target_shape)
            for model in models
        ],
        [2, 3, 1, 4, 0],
    )


def map_dataset_multiple_annotators(
    dataset: Tensor,
    target_shape: tuple[int, int],
    model_shape: tuple[int, int],
    batch_size: int,
    disturbance_models: list[Model],
) -> Tensor:
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
            img,
            add_noisy_annotators(
                tf.expand_dims(img, 0),
                disturbance_models,
                model_shape=model_shape,
                target_shape=target_shape,
            ),
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
