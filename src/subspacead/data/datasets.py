import glob
import os
from pathlib import Path
import numpy as np
from PIL import Image
import pandas as pd
import logging


class BaseDatasetHandler:
    """Abstract base class for dataset handlers."""

    def __init__(self, root_path, category):
        self.root_path = Path(root_path)
        self.category = category
        self.category_path = self.root_path / category

    def get_train_paths(self):
        raise NotImplementedError

    def get_validation_paths(self):
        return []  # Default: no validation set

    def get_test_paths(self):
        raise NotImplementedError

    def get_ground_truth_path(self, test_path: str):
        raise NotImplementedError

    def get_ground_truth_mask(self, test_path: str, res: tuple):
        gt_path_str = self.get_ground_truth_path(test_path)
        if not gt_path_str or not os.path.exists(gt_path_str):
            return np.zeros((res[1], res[0]), dtype=np.uint8)

        mask = (
            Image.open(gt_path_str)
            .convert("L")
            .resize(res, Image.Resampling.NEAREST)  # res is (W, H)
        )
        return (np.array(mask) > 0).astype(np.uint8)  # returns (H, W) array


class MVTecADDataset(BaseDatasetHandler):
    """Handler for the original MVTec AD dataset structure."""

    def get_train_paths(self):
        return sorted(glob.glob(str(self.category_path / "train" / "good" / "*.png")))

    def get_test_paths(self):
        return sorted(glob.glob(str(self.category_path / "test" / "*" / "*.png")))

    def get_ground_truth_path(self, test_path: str):
        p = Path(test_path)
        return str(
            self.category_path / "ground_truth" / p.parent.name / f"{p.stem}_mask.png"
        )


class MVTecLOCODataset(BaseDatasetHandler):
    """
    Handler for MVTec LOCO AD.
    Structure:
        train/good
        validation/good
        test/good, test/logical_anomalies, test/structural_anomalies
        ground_truth/logical_anomalies/000/000.png (nested) OR standard _mask.png
    """

    def get_train_paths(self):
        return sorted(glob.glob(str(self.category_path / "train" / "good" / "*.png")))

    def get_validation_paths(self):
        return sorted(
            glob.glob(str(self.category_path / "validation" / "good" / "*.png"))
        )

    def get_test_paths(self):
        return sorted(
            glob.glob(str(self.category_path / "test" / "**" / "*.png"), recursive=True)
        )

    def get_ground_truth_path(self, test_path: str):
        p = Path(test_path)
        anomaly_type = p.parent.name  # e.g., 'logical_anomalies'

        if anomaly_type == "good":
            return None
        candidate_1 = (
            self.category_path / "ground_truth" / anomaly_type / f"{p.stem}_mask.png"
        )
        if candidate_1.exists():
            return str(candidate_1)
        candidate_2 = (
            self.category_path / "ground_truth" / anomaly_type / p.stem / "000.png"
        )
        if candidate_2.exists():
            return str(candidate_2)
        candidate_3 = (
            self.category_path / "ground_truth" / anomaly_type / p.stem / f"{p.name}"
        )
        if candidate_3.exists():
            return str(candidate_3)

        return None


class MVTecAD2Dataset(BaseDatasetHandler):
    """Handler for the MVTec AD 2 dataset structure."""

    def get_train_paths(self):
        return sorted(glob.glob(str(self.category_path / "train" / "good" / "*.png")))

    def get_validation_paths(self):
        return sorted(
            glob.glob(str(self.category_path / "validation" / "good" / "*.png"))
        )

    def get_test_paths(self):
        return sorted(
            glob.glob(str(self.category_path / "test_public" / "*" / "*.png"))
        )

    def get_ground_truth_path(self, test_path: str):
        p = Path(test_path)
        return str(
            self.category_path
            / "test_public"
            / "ground_truth"
            / p.parent.name
            / f"{p.stem}_mask.png"
        )


class VisADataset(BaseDatasetHandler):
    """Handler for VisA dataset with structure:
    category/
    ├── ground_truth/bad/*.png
    ├── test/{good,bad}/*.JPG
    └── train/good/*.JPG
    """

    def get_train_paths(self):
        return sorted(glob.glob(str(self.category_path / "train" / "good" / "*.JPG")))

    def get_test_paths(self):
        # include both good and bad
        return sorted(glob.glob(str(self.category_path / "test" / "*" / "*.JPG")))

    def get_ground_truth_path(self, test_path: str):
        p = Path(test_path)
        # only bad samples have masks
        if "bad" in p.parts:
            mask_path = self.category_path / "ground_truth" / "bad" / f"{p.stem}.png"
            if mask_path.exists():
                return str(mask_path)
        # good samples have no ground truth
        return None


class CustomDataset(BaseDatasetHandler):
    """Handler for custom datasets with structure:
    category/
    ├── train/good/*.png (or .jpg, .bmp, .jpeg, .tif, .tiff)
    └── test/<any_subfolder>/*.png
    Optional: ground_truth/<defect_type>/<stem>_mask.png
    """

    EXTENSIONS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff",
                  "*.PNG", "*.JPG", "*.JPEG", "*.BMP", "*.TIF", "*.TIFF")

    def _glob_multi(self, pattern_base):
        paths = []
        for ext in self.EXTENSIONS:
            paths.extend(glob.glob(str(pattern_base / ext)))
        return sorted(set(paths))

    def get_train_paths(self):
        return self._glob_multi(self.category_path / "train" / "good")

    def get_test_paths(self):
        paths = []
        for ext in self.EXTENSIONS:
            paths.extend(
                glob.glob(str(self.category_path / "test" / "*" / ext))
            )
        return sorted(set(paths))

    def get_ground_truth_path(self, test_path: str):
        p = Path(test_path)
        # Try MVTec-style mask naming
        gt = self.category_path / "ground_truth" / p.parent.name / f"{p.stem}_mask.png"
        if gt.exists():
            return str(gt)
        # Try same-name mask
        gt2 = self.category_path / "ground_truth" / p.parent.name / p.name
        if gt2.exists():
            return str(gt2)
        return None


def get_dataset_handler(name: str, root_path: str, category: str) -> BaseDatasetHandler:
    """Factory function to get the correct dataset handler."""
    if name == "mvtec_ad":
        return MVTecADDataset(root_path, category)
    elif name == "mvtec_loco":
        return MVTecLOCODataset(root_path, category)
    elif name == "mvtec_ad2":
        return MVTecAD2Dataset(root_path, category)
    elif name == "visa":
        return VisADataset(root_path, category)
    elif name == "custom":
        return CustomDataset(root_path, category)
    else:
        raise ValueError(f"Unknown dataset: {name}")
