import numpy as np
import SimpleITK as sitk
import tensorflow as tf

from seg_tgce.data.oxford_pet.oxford_iiit_pet import OxfordIiitPet
from seg_tgce.data.oxford_pet.oxford_pet import (
    fetch_models,
)
from seg_tgce.data.utils import (
    LabelerAssignmentManager,
    map_dataset_multiple_annotators,
)
from seg_tgce.metrics import DiceCoefficient, JaccardCoefficient

TARGET_SHAPE = (256, 256)
BATCH_SIZE = 64
NUM_CLASSES = 3
NOISE_LEVELS = [-20.0, 0.0, 10.0]
NUM_SCORERS = len(NOISE_LEVELS)
LABELING_RATES = [1.0, 0.7, 0.3]
SEED = 42

MODEL_ORIGINAL_SHAPE = (512, 512)


def perform_staple(masks: tf.Tensor, labeler_masks: tf.Tensor) -> np.ndarray:
    """
    Perform STAPLE algorithm on predictions from multiple annotators.
    Args:
        masks: tensorflow tensor of shape (batch_size, height, width, num_classes, num_scorers)
        labeler_masks: tensorflow tensor indicating which labelers labeled which images
    Returns:
        numpy array of shape (batch_size, height, width, num_classes)
    """
    masks_np = masks.numpy()
    labeler_masks_np = labeler_masks.numpy()

    batch_size, height, width, num_classes, num_scorers = masks_np.shape
    staple_predictions = np.zeros((batch_size, height, width, num_classes))

    for b in range(batch_size):
        active_labelers = np.where(labeler_masks_np[b] == 1)[0]

        for c in range(num_classes):
            segmentations_sitk = []
            sum_of_all_binary_masks_for_class = np.zeros((height, width), dtype=np.uint8)

            for active_labeler in active_labelers:
                class_scores_for_labeler = masks_np[b, :, :, c, active_labeler]
                binary_segmentation = (class_scores_for_labeler > 0.5).astype(np.uint8)
                sum_of_all_binary_masks_for_class += binary_segmentation
                sitk_mask = sitk.GetImageFromArray(binary_segmentation)
                segmentations_sitk.append(sitk_mask)

            if not segmentations_sitk:
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
                staple_filter = sitk.STAPLEImageFilter()  # type: ignore[no-untyped-call]
                staple_filter.SetForegroundValue(1)  # type: ignore[no-untyped-call]

                if (
                    len(segmentations_sitk) < 2
                    and np.sum(sum_of_all_binary_masks_for_class) > 0
                ):
                    staple_predictions[b, :, :, c] = sitk.GetArrayFromImage(
                        segmentations_sitk[0]
                    )
                elif len(segmentations_sitk) >= 2:
                    staple_result = staple_filter.Execute(segmentations_sitk)  # type: ignore[no-untyped-call]
                    staple_result_np = sitk.GetArrayFromImage(staple_result)
                    if np.isnan(np.sum(staple_result_np)):
                        staple_predictions[b, :, :, c] = np.zeros(
                            (height, width), dtype=np.float32
                        )
                    else:
                        staple_predictions[b, :, :, c] = staple_result_np
                else:
                    staple_predictions[b, :, :, c] = np.zeros(
                        (height, width), dtype=np.float32
                    )
            except Exception:
                staple_predictions[b, :, :, c] = np.zeros(
                    (height, width), dtype=np.float32
                )

    return staple_predictions


def evaluate_staple(test_data: tf.data.Dataset) -> tuple[float, float]:
    """
    Evaluate STAPLE algorithm on test data.
    Returns:
        tuple of (average_dice, average_jaccard)
    """
    dice_fn = DiceCoefficient(
        num_classes=NUM_CLASSES,
        name="dice_coefficient",
    )
    jaccard_fn = JaccardCoefficient(
        num_classes=NUM_CLASSES,
        name="jaccard_coefficient",
    )

    total_dice = 0
    total_jaccard = 0
    num_batches = 0
    print(f"Total batches: {len(test_data)}")

    for num, batch in enumerate(test_data):
        print(f"Processing batch {num} of {len(test_data)}")
        images, masks_tensor, labeled_by, ground_truth = batch

        staple_predictions_np = perform_staple(masks_tensor, labeled_by)
        staple_predictions_tf = tf.convert_to_tensor(
            staple_predictions_np, dtype=tf.float32
        )

        dice_score = dice_fn(ground_truth, staple_predictions_tf)
        jaccard_score = jaccard_fn(ground_truth, staple_predictions_tf)

        if not (tf.math.is_nan(dice_score) or tf.math.is_nan(jaccard_score)):
            total_dice += dice_score.numpy()
            total_jaccard += jaccard_score.numpy()
            num_batches += 1

    if num_batches == 0:
        print("Warning: All batches resulted in NaN metrics for STAPLE.")
        return np.nan, np.nan

    avg_dice = total_dice / num_batches
    avg_jaccard = total_jaccard / num_batches

    return avg_dice, avg_jaccard


def main() -> None:
    disturbance_models = fetch_models(NOISE_LEVELS, seed=SEED)

    print("\nSTAPLE Results:")
    print("-" * 50)
    print(f"{'Labeling Rate':<15} {'Dice Coefficient':<20} {'Jaccard Coefficient':<20}")
    print("-" * 50)

    dataset = OxfordIiitPet()
    _, _, test_dataset = dataset()

    for labeling_rate in LABELING_RATES:
        labeler_manager = LabelerAssignmentManager(
            num_samples=len(test_dataset),
            num_labelers=len(disturbance_models),
            labeling_rate=labeling_rate,
            seed=42,
        )
        test = map_dataset_multiple_annotators(
            dataset=test_dataset,
            target_shape=TARGET_SHAPE,
            model_shape=MODEL_ORIGINAL_SHAPE,
            batch_size=BATCH_SIZE,
            disturbance_models=disturbance_models,
            labeler_manager=labeler_manager,
        )

        avg_dice, avg_jaccard = evaluate_staple(test.cache())

        print(f"{labeling_rate:<15.1f} {avg_dice:<20.4f} {avg_jaccard:<20.4f}")


if __name__ == "__main__":
    main()
