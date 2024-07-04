from seg_tgce.data.crowd_seg import get_all_data


def main() -> None:
    train, val, test = get_all_data(batch_size=8, with_sparse_data=True)
    val.visualize_sample(batch_index=138, sample_indexes=[2, 3, 4, 5])
    print(f"Train: {len(train)} batches, {len(train) * train.batch_size} samples")
    print(f"Val: {len(val)} batches, {len(val) * val.batch_size} samples")
    print(f"Test: {len(test)} batches, {len(test) * test.batch_size} samples")


main()
