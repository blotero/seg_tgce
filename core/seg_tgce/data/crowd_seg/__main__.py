from seg_tgce.data.crowd_seg import get_all_data


def main() -> None:
    train, val, test = get_all_data(batch_size=8)
    val.visualize_sample(batch_index=138, sample_indexes=[2, 3, 4, 5])
    print(f"Train: {len(train)}")
    print(f"Val: {len(val)}")
    print(f"Test: {len(test)}")


main()
