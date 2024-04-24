# add unit test for previous function
from pathlib import Path

import pytest

from core.seg_tgce.data.crowd_seg.map import DataTarget, find_annotators_alias

BASE_PATH = Path("/home/brandon/unal/maestria/datasets/Histology Data")


@pytest.mark.with_crowd_seg_data
def test_find_annotators_alias_train():
    base_dir_path = BASE_PATH
    data_target = DataTarget.TRAIN
    annotators_alias = find_annotators_alias(data_target, base_dir_path)
    assert sorted(annotators_alias) == sorted(
        [
            "expert",
            "MV",
            "NP1",
            "NP10",
            "NP11",
            "NP12",
            "NP14",
            "NP15",
            "NP16",
            "NP17",
            "NP18",
            "NP19",
            "NP2",
            "NP20",
            "NP21",
            "NP3",
            "NP4",
            "NP5",
            "NP6",
            "NP7",
            "NP8",
            "NP9",
            "STAPLE",
        ]
    )


@pytest.mark.with_crowd_seg_data
def test_find_annotators_alias_val():
    base_dir_path = BASE_PATH
    data_target = DataTarget.VAL
    annotators_alias = find_annotators_alias(data_target, base_dir_path)
    assert sorted(annotators_alias) == sorted(["NP21", "NP16", "NP8", "expert"])


@pytest.mark.with_crowd_seg_data
def test_find_annotators_alias_test():
    base_dir_path = BASE_PATH
    data_target = DataTarget.TEST
    annotators_alias = find_annotators_alias(data_target, base_dir_path)
    assert annotators_alias == ["expert"]
