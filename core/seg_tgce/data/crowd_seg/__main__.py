from matplotlib import pyplot as plt

from seg_tgce.data.crowd_seg import get_all_data
from seg_tgce.data.crowd_seg.generator import CrowdSegDataGenerator


def main() -> None:
    print("Loading data...")
    train, val, test = get_all_data(batch_size=16)

    # Get a sample batch from each generator
    train_batch = next(iter(train))
    val_batch = next(iter(val))
    test_batch = next(iter(test))

    # Print shapes
    print("\nTrain data shapes:")
    print(f"Images shape: {train_batch[0].shape}")
    print(f"Ground truth mask shape: {train_batch[1].shape}")
    print(f"Labeler masks shape: {train_batch[2].shape}")

    print("\nValidation data shapes:")
    print(f"Images shape: {val_batch[0].shape}")
    print(f"Ground truth mask shape: {val_batch[1].shape}")
    print(f"Labeler masks shape: {val_batch[2].shape}")

    print("\nTest data shapes:")
    print(f"Images shape: {test_batch[0].shape}")
    print(f"Ground truth mask shape: {test_batch[1].shape}")
    print(f"Labeler masks shape: {test_batch[2].shape}")

    fig = val.visualize_sample(batch_index=75, sample_indexes=[0, 1, 4, 5])
    fig.tight_layout()
    fig.savefig(
        "/home/brandon/unal/maestria/master_thesis/Cap1/Figures/multiannotator-segmentation.png"
    )
    print(f"Train: {len(train)} batches, {len(train) * train.batch_size} samples")
    print(f"Val: {len(val)} batches, {len(val) * val.batch_size} samples")
    print(f"Test: {len(test)} batches, {len(test) * test.batch_size} samples")

    print("Loading train data with trimmed scorers...")
    train = CrowdSegDataGenerator(
        batch_size=8,
        trim_n_scorers=6,
    )
    print(f"Train: {len(train)} batches, {len(train) * train.batch_size} samples")
    print(f"Train scorers tags: {train.scorers_tags}")


main()
