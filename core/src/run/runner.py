from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Tuple

from keras.models import Model


@dataclass
class RunningSessionParams:
    n_epochs: int
    target_img_shape: Tuple[int, int]
    batch_size: int
    num_annotators: int
    extra: dict[str, Any] | None
    plotting_frequency: int = 10


@dataclass
class SessionResults:
    models: dict[str, Model]
    train_metadata: dict[str, Any] | None


@dataclass
class SessionPartialResults:
    train_metadata: dict[str, Any]


class Runner(ABC):
    @abstractmethod
    async def run(self) -> SessionResults:
        pass

    @abstractmethod
    def __init__(self, params: RunningSessionParams) -> None:
        pass

    @abstractmethod
    async def stop(self) -> SessionPartialResults:
        pass
