# Inspired from https://github.com/UN-GCPDS/python-gcpds.image_segmentation
# Original license: BSD-2-Clause

from functools import cache
from typing import List, Tuple

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import Tensor


class OxfordIiitPet:
    def __init__(
        self,
        split: Tuple[float, float, float] = (70.0, 15.0, 15.0),
        one_hot: bool = True,
    ):
        self.one_hot = one_hot
        self.split = OxfordIiitPet._get_splits(split)
        dataset, info = tfds.load(
            "oxford_iiit_pet:3.*.*", with_info=True, split=self.split
        )
        self.info = info
        train, val, test = dataset
        self.classes = 3
        train = train.map(self._keep_interface)
        val = val.map(self._keep_interface)
        test = test.map(self._keep_interface)

        self.mapped_dataset = train, val, test
        self.labels_info = {0: "cat", 1: "dog"}

    @cache  # pylint: disable=method-cache-max-size-none
    def load_instance_by_id(  # type: ignore
        self,
        id_img: str,
    ) -> Tuple[Tensor, Tensor, Tensor, str]:
        for dataset in self.mapped_dataset:
            dataset = dataset.filter(
                lambda img, mask, label, id_image: id_image == id_img
            )
            for x in dataset:
                return x

    @staticmethod
    def _get_splits(splits: Tuple[float, float, float]) -> List[str]:
        percentage_sum = 0.0
        splits_ = []
        for percentage in splits:
            percentage_sum += percentage
            splits_.append(f"train[{percentage_sum-percentage}%:{percentage_sum}%]")
        return splits_

    def to_one_hot(self, mask: Tensor) -> Tensor:
        one_hot = tf.one_hot(mask, self.classes)
        return tf.gather(one_hot, 0, axis=2)

    def _keep_interface(self, x: dict) -> Tuple[Tensor, Tensor, Tensor, str]:
        img = tf.cast(x["image"], tf.float32) / 255.0
        mask = x["segmentation_mask"] - 1
        mask = self.to_one_hot(mask) if self.one_hot else mask
        label = x["species"]
        id_image = x["file_name"]
        return img, mask, label, id_image

    def __call__(
        self,
    ):
        return self.mapped_dataset


if __name__ == "__main__":
    dataset = OxfordIiitPet()
    train_dataset, val_dataset, test_dataset = dataset()
    for img, mask, label, id_img in train_dataset.take(1):
        print(img.shape, mask.shape, label, id_img)
        break
    for img, mask, label, id_img in val_dataset.take(1):
        print(img.shape, mask.shape, label, id_img)
        break
    for img, mask, label, id_img in test_dataset.take(1):
        print(img.shape, mask.shape, label, id_img)
        break
