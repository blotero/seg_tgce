from typing import Tuple

from .generator import CrowdSegDataGenerator, DataSchema
from .stage import Stage

DEFAULT_TARGET_SIZE = (512, 512)


def get_all_data(
    image_size: Tuple[int, int] = DEFAULT_TARGET_SIZE,
    batch_size: int = 32,
    shuffle: bool = False,
    with_sparse_data: bool = False,
    trim_n_scorers: int | None = None,
) -> Tuple[CrowdSegDataGenerator, ...]:
    """
    Retrieve all data generators for the crowd segmentation task.
    returns a tuple of ImageDataGenerator instances for the train, val, and test stages.
    """
    return tuple(
        CrowdSegDataGenerator(
            batch_size=batch_size,
            image_size=image_size,
            shuffle=shuffle,
            stage=stage,
            schema=DataSchema.MA_SPARSE if with_sparse_data else DataSchema.MA_RAW,
            trim_n_scorers=trim_n_scorers,
        )
        for stage in (Stage.TRAIN, Stage.VAL, Stage.TEST)
    )


def get_stage_data(
    stage: Stage,
    image_size: Tuple[int, int] = DEFAULT_TARGET_SIZE,
    batch_size: int = 32,
    shuffle: bool = False,
    with_sparse_data: bool = False,
) -> CrowdSegDataGenerator:
    """
    Retrieve a data generator for a specific stage of the crowd segmentation task.
    """
    return CrowdSegDataGenerator(
        batch_size=batch_size,
        image_size=image_size,
        shuffle=shuffle,
        stage=stage,
        schema=DataSchema.MA_SPARSE if with_sparse_data else DataSchema.MA_RAW,
    )
