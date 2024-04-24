from enum import Enum
from pathlib import Path


class DataTarget(Enum):
    TRAIN = "Train"
    VAL = "Val"
    TEST = "Test"


MASKS_SUB_DIR = "masks"
PATCHES_SUB_DIR = "patches"


def find_annotators_alias(data_target: DataTarget, base_dir_path: Path) -> list[str]:
    data_target_path = base_dir_path / MASKS_SUB_DIR / data_target.value
    annotators_alias = [
        path.name for path in data_target_path.iterdir() if path.is_dir()
    ]
    return annotators_alias


def produce_tf_dataset(
    data_target: DataTarget, base_dir_path: Path, annotator_alias: str
) -> Path:
    data_target_path = base_dir_path / MASKS_SUB_DIR / data_target.value
    tf_image_folder = data_target_path / annotator_alias
    return tf_image_folder
