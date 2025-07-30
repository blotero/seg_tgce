import tensorflow as tf

from seg_tgce.data.oxford_pet.oxford_pet_tfds import get_data_multiple_annotators_tfds
from seg_tgce.loss.tgce import TgceFeatures, TgcePixel, TgceScalar

if __name__ == "__main__":
    NOISE_LEVELS_SNR = [-20.0, 20.0]
    TARGET_SHAPE = (128, 128)
    BATCH_SIZE = 16
    NUM_CLASSES = 3

    train_data, val_data, test_data = get_data_multiple_annotators_tfds(
        noise_levels_snr=NOISE_LEVELS_SNR,
        target_shape=TARGET_SHAPE,
        batch_size=BATCH_SIZE,
        labeling_rate=0.7,
    )
    scalar_loss = TgceScalar(num_classes=NUM_CLASSES)
    features_loss = TgceFeatures(num_classes=NUM_CLASSES)
    pixel_loss = TgcePixel(num_classes=NUM_CLASSES)

    for i, data in enumerate(train_data):
        ground_truth = data["ground_truth"]
        images = data["image"]
        masks = data["masks"]
        labeler_mask = data["labeler_mask"]
        print(ground_truth.shape)
        print(images.shape)
        print(masks.shape)
        print(labeler_mask.shape)
        # lambda r with zeroes
        scalar_lambda_r = tf.ones((BATCH_SIZE, len(NOISE_LEVELS_SNR)))
        features_lambda_r = tf.ones((BATCH_SIZE, 4, 4, len(NOISE_LEVELS_SNR)))
        pixel_lambda_r = tf.ones((BATCH_SIZE, *TARGET_SHAPE, len(NOISE_LEVELS_SNR)))
        y_pred = tf.random.normal((BATCH_SIZE, *TARGET_SHAPE, NUM_CLASSES))

        scalar_loss_value = scalar_loss.call(masks, y_pred, scalar_lambda_r, labeler_mask)
        features_loss_value = features_loss.call(
            masks, y_pred, features_lambda_r, labeler_mask
        )
        pixel_loss_value = pixel_loss.call(masks, y_pred, pixel_lambda_r, labeler_mask)

        break
