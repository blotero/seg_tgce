from typing import Tuple

from .generator import ImageDataGenerator
from .stage import Stage


def get_all_data(
    image_size: Tuple[int, int] = (256, 256),
    batch_size: int = 32,
    shuffle: bool = False,
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
        )
        for stage in (Stage.TRAIN, Stage.VAL, Stage.TEST)
    )


def get_stage_data(
    stage: Stage,
    image_size: Tuple[int, int] = (256, 256),
    batch_size: int = 32,
    shuffle: bool = False,
) -> ImageDataGenerator:
    """
    Retrieve a data generator for a specific stage of the crowd segmentation task.
    """
    return ImageDataGenerator(
        batch_size=batch_size,
        image_size=image_size,
        shuffle=shuffle,
        stage=stage,
    )
