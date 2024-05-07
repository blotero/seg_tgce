import json
from dataclasses import dataclass


@dataclass
class ModelResult:
    losses: list[float]
    dices: list[float]
    val_losses: list[float]
    val_dices: list[float]

    def save_as_json(self, filename: str) -> None:
        dataclass_dict = self.__dict__
        json_str = json.dumps(dataclass_dict, indent=4)
        with open(filename, "w", encoding="utf-8") as f:
            f.write(json_str)
