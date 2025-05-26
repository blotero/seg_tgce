from matplotlib import pyplot as plt

from seg_tgce.data.crowd_seg import get_crowd_seg_data


def main() -> None:
    print("Loading data...")
    train, val, test = get_crowd_seg_data(batch_size=128)

    fig = train.visualize_sample(
        batch_index=6, sample_indexes=[0, 1, 30, 31, 63, 64, 126, 127]
    )
    fig.tight_layout()
    # fig.savefig(
    # "/home/brandon/unal/maestria/master_thesis/Cap1/Figures/multiannotator-segmentation.png"
    # )
    plt.show()
    print(f"Train: {len(train)} batches, {len(train) * train.batch_size} samples")
    print(f"Val: {len(val)} batches, {len(val) * val.batch_size} samples")
    print(f"Test: {len(test)} batches, {len(test) * test.batch_size} samples")


main()
