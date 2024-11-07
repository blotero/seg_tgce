from seg_tgce.data.crowd_seg import get_all_data


def main() -> None:
    train, val, test = get_all_data(batch_size=2, with_sparse_data=False)
    print("Processing train")

    for db in (train, val, test):
        db.populate_metadata()
        db.store_metadata()


if __name__ == "__main__":
    main()
