import numpy as np
import SimpleITK as sitk
import tensorflow as tf
from seg_tgce.data.oxford_pet.oxford_pet import (
    fetch_models,
    get_data_multiple_annotators,
)
from seg_tgce.metrics import DiceCoefficient, JaccardCoefficient

TARGET_SHAPE = (256, 256)
GROUND_TRUTH_INDEX = -1  # Last labeler is ground truth
BATCH_SIZE = 64
NUM_CLASSES = 3
NOISE_LEVELS = [-20.0, 0.0, 10.0]
NUM_SCORERS = len(NOISE_LEVELS)
LABELING_RATES = [0.3]


def perform_staple(masks, labeler_masks):
    """
    Perform STAPLE algorithm on predictions from multiple annotators,
    excluding the ground truth labeler from STAPLE input.
    Args:
        masks: tensorflow tensor of shape (batch_size, height, width, num_classes, num_scorers)
        labeler_masks: tensorflow tensor indicating which labelers labeled which images
    Returns:
        numpy array of shape (batch_size, height, width, num_classes)
    """
    # Convert tensors to numpy arrays
    masks_np = masks.numpy()
    labeler_masks_np = labeler_masks.numpy()

    batch_size, height, width, num_classes, num_scorers = masks_np.shape
    staple_predictions = np.zeros((batch_size, height, width, num_classes))

    # Determine the actual index of the ground truth scorer
    # (num_scorers - 1) because GROUND_TRUTH_INDEX is -1
    gt_scorer_actual_idx = num_scorers - 1

    # For each image in the batch
    for b in range(batch_size):
        # Get which labelers labeled this image
        active_labelers_all = np.where(labeler_masks_np[b] == 1)[0]

        # Exclude the ground truth scorer from the list of labelers for STAPLE input
        labelers_for_staple_input = [
            l_idx for l_idx in active_labelers_all if l_idx != gt_scorer_actual_idx
        ]

        # For each class
        for c in range(num_classes):
            segmentations_sitk = []
            sum_of_all_binary_masks_for_class = np.zeros(
                (height, width), dtype=np.uint8
            )

            # Get binary segmentation for each active labeler (excluding GT)
            for l in labelers_for_staple_input:
                class_scores_for_labeler = masks_np[b, :, :, c, l]
                binary_segmentation = (class_scores_for_labeler > 0.5).astype(np.uint8)
                sum_of_all_binary_masks_for_class += binary_segmentation
                sitk_mask = sitk.GetImageFromArray(binary_segmentation)
                segmentations_sitk.append(sitk_mask)

            if not segmentations_sitk:  # No non-GT active labelers for this image
                staple_predictions[b, :, :, c] = np.zeros(
                    (height, width), dtype=np.float32
                )
                continue

            if np.sum(sum_of_all_binary_masks_for_class) == 0:
                staple_predictions[b, :, :, c] = np.zeros(
                    (height, width), dtype=np.float32
                )
                continue

            try:
                staple_filter = sitk.STAPLEImageFilter()
                staple_filter.SetForegroundValue(1)

                if (
                    len(segmentations_sitk) < 2
                    and np.sum(sum_of_all_binary_masks_for_class) > 0
                ):  # If only one non-GT labeler, their segmentation is the result
                    staple_predictions[b, :, :, c] = sitk.GetArrayFromImage(
                        segmentations_sitk[0]
                    )
                elif len(segmentations_sitk) >= 2:
                    staple_result = staple_filter.Execute(segmentations_sitk)
                    staple_result_np = sitk.GetArrayFromImage(staple_result)
                    if np.isnan(np.sum(staple_result_np)):
                        staple_predictions[b, :, :, c] = np.zeros(
                            (height, width), dtype=np.float32
                        )
                    else:
                        staple_predictions[b, :, :, c] = staple_result_np
                else:  # No segmentations with foreground, or fewer than 1 segmentation (already handled by checks above)
                    staple_predictions[b, :, :, c] = np.zeros(
                        (height, width), dtype=np.float32
                    )
            except Exception as e:
                # print(f"Error during STAPLE for batch {b}, class {c}: {e}") # Optional debug
                staple_predictions[b, :, :, c] = np.zeros(
                    (height, width), dtype=np.float32
                )

    return staple_predictions


def evaluate_staple(test_data):
    """
    Evaluate STAPLE algorithm on test data.
    Returns:
        tuple of (average_dice, average_jaccard)
    """
    # Initialize metrics
    dice_fn = DiceCoefficient(
        num_classes=NUM_CLASSES,
        name="dice_coefficient",
        ground_truth_index=GROUND_TRUTH_INDEX,  # This will use the last scorer from 'masks' as GT
    )
    jaccard_fn = JaccardCoefficient(
        num_classes=NUM_CLASSES,
        name="jaccard_coefficient",
        ground_truth_index=GROUND_TRUTH_INDEX,  # This will use the last scorer from 'masks' as GT
    )

    # Process test data
    total_dice = 0
    total_jaccard = 0
    num_batches = 0
    print(f"Total batches: {len(test_data)}")

    for num, batch in enumerate(test_data):
        print(f"Processing batch {num} of {len(test_data)}")
        images, masks_tensor, labeler_masks_tensor = batch

        # Perform STAPLE
        staple_predictions_np = perform_staple(masks_tensor, labeler_masks_tensor)
        staple_predictions_tf = tf.convert_to_tensor(
            staple_predictions_np, dtype=tf.float32
        )

        dice_score = dice_fn(masks_tensor, staple_predictions_tf)
        jaccard_score = jaccard_fn(masks_tensor, staple_predictions_tf)

        if not (tf.math.is_nan(dice_score) or tf.math.is_nan(jaccard_score)):
            total_dice += dice_score.numpy()
            total_jaccard += jaccard_score.numpy()
            num_batches += 1

    if num_batches == 0:
        print("Warning: All batches resulted in NaN metrics for STAPLE.")
        return np.nan, np.nan

    # Calculate average metrics
    avg_dice = total_dice / num_batches
    avg_jaccard = total_jaccard / num_batches

    return avg_dice, avg_jaccard


def main():
    # Fetch the disturbance models
    disturbance_models = fetch_models(NOISE_LEVELS)

    print("\nSTAPLE Results:")
    print("-" * 50)
    print(f"{'Labeling Rate':<15} {'Dice Coefficient':<20} {'Jaccard Coefficient':<20}")
    print("-" * 50)

    for labeling_rate in LABELING_RATES:
        # Get the data for current labeling rate
        _, _, test = get_data_multiple_annotators(
            annotation_models=disturbance_models,
            target_shape=TARGET_SHAPE,
            batch_size=BATCH_SIZE,
            labeling_rate=labeling_rate,
        )

        # Evaluate STAPLE
        avg_dice, avg_jaccard = evaluate_staple(test.cache())

        # Print results
        print(f"{labeling_rate:<15.1f} {avg_dice:<20.4f} {avg_jaccard:<20.4f}")


if __name__ == "__main__":
    main()
