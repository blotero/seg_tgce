from typing import Tuple

from .generator import DataSchema, ImageDataGenerator
from .stage import Stage

DEFAULT_TARGET_SIZE = (512, 512)


def get_all_data(
    image_size: Tuple[int, int] = DEFAULT_TARGET_SIZE,
    batch_size: int = 32,
    shuffle: bool = False,
    with_sparse_data: bool = False,
) -> Tuple[ImageDataGenerator, ...]:
    """
    Retrieve all data generators for the crowd segmentation task.
    returns a tuple of ImageDataGenerator instances for the train, val, and test stages.
    """
    return tuple(
        ImageDataGenerator(
            batch_size=batch_size,
            image_size=image_size,
            shuffle=shuffle,
            stage=stage,
            schema=DataSchema.MA_SPARSE if with_sparse_data else DataSchema.MA_RAW,
        )
        for stage in (Stage.TRAIN, Stage.VAL, Stage.TEST)
    )


def get_stage_data(
    stage: Stage,
    image_size: Tuple[int, int] = DEFAULT_TARGET_SIZE,
    batch_size: int = 32,
    shuffle: bool = False,
    with_sparse_data: bool = False,
) -> ImageDataGenerator:
    """
    Retrieve a data generator for a specific stage of the crowd segmentation task.
    """
    return ImageDataGenerator(
        batch_size=batch_size,
        image_size=image_size,
        shuffle=shuffle,
        stage=stage,
        schema=DataSchema.MA_SPARSE if with_sparse_data else DataSchema.MA_RAW,
    )
