from typing import Tuple

from .generator import ImageDataGenerator
from .stage import Stage

_DEFAULT_N_CLASSES = 6


def get_all_data(
    n_classes: int = _DEFAULT_N_CLASSES,
    image_size: Tuple[int, int] = (256, 256),
    batch_size: int = 32,
    shuffle: bool = True,
) -> Tuple[ImageDataGenerator, ...]:
    """
    Retrieve all data generators for the crowd segmentation task.
    returns a tuple of ImageDataGenerator instances for the train, val, and test stages.
    """
    return tuple(
        ImageDataGenerator(
            batch_size=batch_size,
            n_classes=n_classes,
            image_size=image_size,
            shuffle=shuffle,
            stage=stage,
        )
        for stage in (Stage.TRAIN, Stage.VAL, Stage.TEST)
    )


def get_stage_data(
    stage: Stage,
    n_classes: int = _DEFAULT_N_CLASSES,
    image_size: Tuple[int, int] = (256, 256),
    batch_size: int = 32,
    shuffle: bool = True,
) -> ImageDataGenerator:
    """
    Retrieve a data generator for a specific stage of the crowd segmentation task.
    """
    return ImageDataGenerator(
        batch_size=batch_size,
        n_classes=n_classes,
        image_size=image_size,
        shuffle=shuffle,
        stage=stage,
    )
