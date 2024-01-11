from typing import Any, List, Tuple

import numpy as np

from .bounding_box import calculate_iou, format_bounding_box


# pylint: disable=too-many-arguments, too-many-locals
def overlay_arrays(
    array_shape: Tuple[int, ...],
    input_arrays: np.ndarray[Any, np.dtype[np.float64]],
    input_labels: np.ndarray[Any, np.dtype[np.float64]],
    num_input_arrays_to_overlay: int,
    max_array_value: int,
    max_iou: float = 0.2,
) -> Tuple[
    np.ndarray[Any, np.dtype[np.float64]],
    np.ndarray[Any, np.dtype[np.float64]],
    np.ndarray[Any, np.dtype[np.float64]],
    np.ndarray[Any, np.dtype[np.float64]],
]:
    output_array = np.zeros(array_shape)

    indices = np.random.randint(
        len(input_arrays), size=num_input_arrays_to_overlay)
    bounding_boxes: List[Any] = []
    bounding_boxes_as_tuple = []
    indices_overlaid = []
    for i in indices:
        bounding_box = overlay_at_random(
            array1=output_array,
            array2=input_arrays[i],
            max_array_value=max_array_value,
            bounding_boxes=bounding_boxes,
            max_iou=max_iou,
        )

        if bounding_box is None:
            break

        indices_overlaid.append(i)

        bounding_boxes_as_tuple.append(
            format_bounding_box(bounding_box, output_type="tuple")
        )
        bounding_boxes.append(bounding_box)

    arrays_overlaid = input_arrays[indices_overlaid]
    labels_overlaid = input_labels[indices_overlaid]
    bounding_boxes_overlaid = np.stack(bounding_boxes_as_tuple)  # type: ignore

    return output_array, arrays_overlaid, labels_overlaid, bounding_boxes_overlaid


def overlay_at_random(
    array1: np.ndarray[Any, np.dtype[np.float64]],
    array2: np.ndarray[Any, np.dtype[np.float64]],
    max_array_value: int,
    bounding_boxes: List[dict[Any, Any]] | None = None,
    max_iou: float = 0.2,
) -> dict[str, int] | None:
    if not bounding_boxes:
        bounding_boxes = []

    height1, width1, *_ = array1.shape
    height2, width2, *_ = array2.shape

    max_x = width1 - width2
    max_y = height1 - height2

    is_valid = False
    max_attempts = 1000
    attempt = 0
    while not is_valid:
        if attempt > max_attempts:
            return None
        attempt += 1
        x = np.random.randint(max_x + 1)
        y = np.random.randint(max_y + 1)

        candidate_bounding_box = {
            "xmin": x,
            "ymin": y,
            "xmax": x + width2,
            "ymax": y + height2,
        }

        is_valid = True
        for bounding_box in bounding_boxes:
            if calculate_iou(bounding_box, candidate_bounding_box) > max_iou:
                is_valid = False
                break

    overlay_array(
        array1=array1, array2=array2, x=x, y=y, max_array_value=max_array_value
    )

    return candidate_bounding_box


def overlay_array(
    array1: np.ndarray[Any, np.dtype[np.float64]],
    array2: np.ndarray[Any, np.dtype[np.float64]],
    x: int,
    y: int,
    max_array_value: int | None = None,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    height1, width1, *other1 = array1.shape
    height2, width2, *other2 = array2.shape

    if height2 > height1 or width2 > width1:
        raise ValueError("array2 must have a smaller shape than array1")

    if other1 != other2:
        raise ValueError(
            "array1 and array2 must have same dimensions beyond dimension 2."
        )

    array1[y: y + height2, x: x + width2, ...] = (
        array1[y: y + height2, x: x + width2, ...] + array2
    )

    array1 = np.clip(array1, a_min=0, a_max=max_array_value, out=array1)

    return array1
