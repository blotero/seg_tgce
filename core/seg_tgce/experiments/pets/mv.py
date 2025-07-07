import numpy as np
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


def boyer_moore_majority_vote(votes):
    """
    Boyer-Moore majority vote algorithm (MJRTY).
    Args:
        votes: array of votes
    Returns:
        majority class
    """
    majority = None
    count = 0

    for vote in votes:
        if count == 0:
            majority = vote
        if vote == majority:
            count += 1
        else:
            count -= 1

    return majority


def perform_majority_voting(masks: np.ndarray, labeler_masks: np.ndarray) -> np.ndarray:
    """
    Perform majority voting on predictions from multiple annotators using vectorized operations.
    Args:
        masks: numpy array of shape (batch_size, height, width, num_classes, num_scorers)
        labeler_masks: numpy array indicating which labelers labeled which images
    Returns:
        numpy array of shape (batch_size, height, width, num_classes)
    """
    class_predictions = np.argmax(masks, axis=3)

    expanded_labeler_masks = np.expand_dims(np.expand_dims(labeler_masks, axis=1), axis=2)

    valid_predictions = class_predictions * expanded_labeler_masks

    majority_votes = np.zeros((*class_predictions.shape[:3], NUM_CLASSES))

    for c in range(NUM_CLASSES):
        votes = (valid_predictions == c).astype(int)
        vote_counts = np.sum(votes, axis=-1)
        majority_votes[..., c] = (vote_counts > 0) & (
            vote_counts >= np.max(vote_counts, axis=-1, keepdims=True)
        )

    return majority_votes


def evaluate_majority_voting(test_data: tf.data.Dataset) -> tuple[float, float]:
    """
    Evaluate majority voting on test data.
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
        images, masks, labeled_by, ground_truth = batch

        mv_predictions = perform_majority_voting(masks, labeled_by)

        dice_score = dice_fn(ground_truth, mv_predictions)
        jaccard_score = jaccard_fn(ground_truth, mv_predictions)

        total_dice += dice_score
        total_jaccard += jaccard_score
        num_batches += 1

    avg_dice = total_dice / num_batches
    avg_jaccard = total_jaccard / num_batches

    return avg_dice, avg_jaccard


def main() -> None:
    disturbance_models = fetch_models(NOISE_LEVELS, seed=SEED)

    print("\nMajority Voting Results:")
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

        avg_dice, avg_jaccard = evaluate_majority_voting(test.cache())

        print(f"{labeling_rate:<15.1f} {avg_dice:<20.4f} {avg_jaccard:<20.4f}")


if __name__ == "__main__":
    main()
