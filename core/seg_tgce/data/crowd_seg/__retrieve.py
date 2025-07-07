import logging
import os
import zipfile

import boto3
from botocore import UNSIGNED
from botocore.client import Config

from seg_tgce.data.crowd_seg.types import Stage

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

_TARGET_DIR = "__data__/crowd_seg"
_BUCKET_NAME = "crowd-seg-data"
PATCHES_OBJECT_NAME = "patches_refined.zip"
MASKS_OBJECT_NAME = "masks_refined.zip"


s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))


def get_masks_dir(stage: Stage) -> str:
    return os.path.join(_TARGET_DIR, "masks", stage.capitalize())


def get_patches_dir(stage: Stage) -> str:
    return os.path.join(_TARGET_DIR, "patches", stage.capitalize())


def _unzip_dirs() -> None:
    logging.info("Unzipping files...")
    for root, _, files in os.walk(_TARGET_DIR):
        for file in files:
            if file.endswith(".zip"):
                with zipfile.ZipFile(os.path.join(root, file), "r") as zip_ref:
                    zip_ref.extractall(root)
                    os.remove(os.path.join(root, file))


class DownloadError(Exception):
    pass


def _fetch_from_s3(obj: str) -> None:
    LOGGER.info("Downloading %s...", obj)
    s3.download_file(
        _BUCKET_NAME,
        obj,
        os.path.join(_TARGET_DIR, obj),
    )


def verify_path(path: str, with_raise: bool = False) -> bool:
    if not os.path.exists(path):
        message = f"Path {path} does not exist."
        if with_raise:
            raise DownloadError(message)
        LOGGER.warning(message)
        return False
    return True


def fetch_data() -> None:
    stages: tuple[Stage, ...] = ("train", "val", "test")
    paths_to_verify = [get_patches_dir(stage) for stage in stages] + [
        get_masks_dir(stage) for stage in stages
    ]
    if all(verify_path(path) for path in paths_to_verify):
        return
    os.makedirs(_TARGET_DIR, exist_ok=True)
    LOGGER.info("Downloading data...")
    for obj in (PATCHES_OBJECT_NAME, MASKS_OBJECT_NAME):
        _fetch_from_s3(obj)
    _unzip_dirs()
    _ = (verify_path(path, with_raise=True) for path in paths_to_verify)
