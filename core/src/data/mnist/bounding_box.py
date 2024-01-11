from typing import Any, Union

import numpy as np


# pylint: disable=too-many-branches
def format_bounding_box(
    bounding_box: Union[
        tuple[float, float, float, float],
        dict[Any, Any],
        np.ndarray[Any, np.dtype[np.float64]],
        list[Any],
    ],
    input_format: str | None = None,
    output_format: str = "xyxy",
    output_type: str = "dict",
) -> Union[dict[Any, Any], tuple[Any]]:
    if output_format == "xyxy":
        if isinstance(bounding_box, dict):
            if all(key in bounding_box for key in ["xmin", "ymin", "xmax", "ymax"]):
                return_value = {
                    "xmin": bounding_box["xmin"],
                    "ymin": bounding_box["ymin"],
                    "xmax": bounding_box["xmax"],
                    "ymax": bounding_box["ymax"],
                }
            elif all(
                key in bounding_box for key in ["xmin", "ymin", "width", "height"]
            ):
                return_value = {
                    "xmin": bounding_box["xmin"],
                    "ymin": bounding_box["ymin"],
                    "xmax": bounding_box["xmin"] + bounding_box["width"],
                    "ymax": bounding_box["ymin"] + bounding_box["height"],
                }
            elif all(key in bounding_box for key in ["x", "y", "width", "height"]):
                return_value = {
                    "xmin": bounding_box["x"],
                    "ymin": bounding_box["y"],
                    "xmax": bounding_box["x"] + bounding_box["width"],
                    "ymax": bounding_box["y"] + bounding_box["height"],
                }
            else:
                raise ValueError(
                    "Incorrect format for bounding_box dictionary. "
                    f"Received: {bounding_box}"
                )
        else:
            if input_format == "xyxy":
                return_value = {
                    "xmin": bounding_box[0],
                    "ymin": bounding_box[1],
                    "xmax": bounding_box[2],
                    "ymax": bounding_box[3],
                }
            elif input_format == "xywh":
                return_value = {
                    "xmin": bounding_box[0],
                    "ymin": bounding_box[1],
                    "xmax": bounding_box[0] + bounding_box[2],
                    "ymax": bounding_box[1] + bounding_box[3],
                }
            else:
                raise ValueError(
                    "If bounding_box is not a dictionary, "
                    'input_format must be specified: "xyxy" or "xywh"'
                )

    elif output_format == "xywh":
        if isinstance(bounding_box, dict):
            if all(key in bounding_box for key in ["xmin", "ymin", "width", "height"]):
                return_value = {
                    "x": bounding_box["xmin"],
                    "y": bounding_box["ymin"],
                    "width": bounding_box["width"],
                    "height": bounding_box["height"],
                }
            elif all(key in bounding_box for key in ["xmin", "ymin", "xmax", "ymax"]):
                return_value = {
                    "x": bounding_box["xmin"],
                    "y": bounding_box["ymin"],
                    "width": bounding_box["xmax"] - bounding_box["xmin"],
                    "height": bounding_box["ymax"] - bounding_box["ymin"],
                }
            elif all(key in bounding_box for key in ["x", "y", "width", "height"]):
                return_value = {
                    "x": bounding_box["x"],
                    "y": bounding_box["y"],
                    "width": bounding_box["width"],
                    "height": bounding_box["height"],
                }
            else:
                raise ValueError(
                    "Incorrect format for bounding_box dictionary. "
                    f"Received: {bounding_box}"
                )
        else:
            if input_format == "xyxy":
                return_value = {
                    "x": bounding_box[0],
                    "y": bounding_box[1],
                    "width": bounding_box[2] - bounding_box[0],
                    "height": bounding_box[3] - bounding_box[1],
                }
            elif input_format == "xywh":
                return_value = {
                    "x": bounding_box[0],
                    "y": bounding_box[1],
                    "width": bounding_box[2],
                    "height": bounding_box[3],
                }
            else:
                raise ValueError(
                    "If bounding_box is not a dictionary, "
                    'input_format must be specified: "xyxy" or "xywh"'
                )
    else:
        raise ValueError(
            f'output_format must be either "xyxy" or "xywh". Received {output_format}'
        )

    if output_type == "tuple":
        return tuple(return_value.values())
    if output_type == "dict":
        return return_value
    raise ValueError(
        f'output_type must be either "dict" or "tuple". Received {output_type}'
    )


def calculate_iou(
    bounding_box1: dict[Any, Any], bounding_box2: dict[Any, Any]
) -> float:
    a1: float = (bounding_box1["xmax"] - bounding_box1["xmin"]) * (
        bounding_box1["ymax"] - bounding_box1["ymin"]
    )
    a2: float = (bounding_box2["xmax"] - bounding_box2["xmin"]) * (
        bounding_box2["ymax"] - bounding_box2["ymin"]
    )

    xmin: float = max(bounding_box1["xmin"], bounding_box2["xmin"])
    ymin: float = max(bounding_box1["ymin"], bounding_box2["ymin"])
    xmax: float = min(bounding_box1["xmax"], bounding_box2["xmax"])
    ymax: float = min(bounding_box1["ymax"], bounding_box2["ymax"])

    if ymin >= ymax or xmin >= xmax:
        return 0

    return ((xmax - xmin) * (ymax - ymin)) / (a1 + a2)
