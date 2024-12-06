from seg_tgce.data.crowd_seg import get_all_data
from seg_tgce.data.crowd_seg.generator import ImageDataGenerator


def main() -> None:
    print("Loading data...")
    train, val, test = get_all_data(batch_size=8)
    val.visualize_sample(batch_index=138, sample_indexes=[2, 3, 4, 5])
    print(f"Train: {len(train)} batches, {len(train) * train.batch_size} samples")
    print(f"Val: {len(val)} batches, {len(val) * val.batch_size} samples")
    print(f"Test: {len(test)} batches, {len(test) * test.batch_size} samples")

    print("Loading train data with trimmed scorers...")
    train = ImageDataGenerator(
        batch_size=8,
        trim_n_scorers=6,
    )
    print(f"Train: {len(train)} batches, {len(train) * train.batch_size} samples")


main()
