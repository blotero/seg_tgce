from enum import Enum


class Stage(Enum):
    """
    Enum class for the stage of the data generator.
    """

    TRAIN = "Train"
    VAL = "Val"
    TEST = "Test"
