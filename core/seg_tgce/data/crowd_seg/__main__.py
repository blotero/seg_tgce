from seg_tgce.data.crowd_seg import get_all_data


def main() -> None:
    train, val, test = get_all_data()
    val.visualize_sample(["NP8", "NP16", "NP21", "expert"])
    print(f"Train: {len(train)}")
    print(f"Val: {len(val)}")
    print(f"Test: {len(test)}")


main()
