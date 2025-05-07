from seg_tgce.data.crowd_seg import get_all_data
from seg_tgce.data.crowd_seg.generator import ImageDataGenerator


def main() -> None:
    print("Loading data...")
    train, val, test = get_all_data(batch_size=16)
    fig = val.visualize_sample(
        batch_index=75, sample_indexes=[2, 5, 8, 15], scorers=["NP8", "expert"]
    )
    fig.tight_layout()
    fig.savefig(
        "/home/brandon/unal/maestria/master_thesis/Cap1/Figures/multiannotator-segmentation.png"
    )
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
