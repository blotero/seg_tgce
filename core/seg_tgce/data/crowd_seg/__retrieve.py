import logging
import os
import zipfile

import gdown

from .stage import Stage

_DATA_URL = "https://drive.google.com/drive/folders/17VukoKpwZclRrDcWSK1aYd_lPeqWNM8N?usp=sharing="
TARGET_DIR = "__data__/crowd_seg"


def get_masks_dir(stage: Stage) -> str:
    return os.path.join(TARGET_DIR, "masks", stage.value)


def get_patches_dir(stage: Stage) -> str:
    return os.path.join(TARGET_DIR, "patches", stage.value)


def unzip_dirs() -> None:
    for root, _, files in os.walk(TARGET_DIR):
        for file in files:
            if file.endswith(".zip"):
                with zipfile.ZipFile(os.path.join(root, file), "r") as zip_ref:
                    zip_ref.extractall(root)
                    os.remove(os.path.join(root, file))


def fetch_data() -> None:
    if not os.path.exists(TARGET_DIR):
        logging.info("Downloading data...")
        gdown.download_folder(_DATA_URL, quiet=False, output=TARGET_DIR)
        unzip_dirs()
        return
    logging.info("Data already exists.")
