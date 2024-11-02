"""Base dataset loader implementation."""

import logging
from pathlib import Path
from typing import Tuple, Union
import os

import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T
from torchvision.datasets.utils import download_and_extract_archive, check_integrity

logger = logging.getLogger(__name__)


class BaseDataset(torch.utils.data.Dataset):
    """Base dataset class with image preprocessing capabilities."""

    def __init__(
        self,
        image_paths: np.ndarray,
        labels: torch.Tensor,
        transform: T.Compose,
        resize_size: int = 232,
        force_preprocess: bool = False,
    ):
        """Initialize dataset.

        Args:
            image_paths: Array of image paths
            labels: Tensor of labels
            transform: Transformations to apply
            resize_size: Size to resize images to
            force_preprocess: Whether to force preprocessing
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.resize_size = resize_size

        # Setup directories
        self.base_dir = Path(os.path.dirname(image_paths[0]))
        self.processed_dir = self.base_dir.parent / f"resize{resize_size}_RGB_png"
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        if force_preprocess or not self._check_processed_images():
            self._preprocess_images()

    def _check_processed_images(self) -> bool:
        """Check if all preprocessed images exist."""
        for img_path in self.image_paths:
            processed_path = self._get_processed_path(img_path)
            if not processed_path.exists():
                return False
        return True

    def _get_processed_path(self, original_path: Union[str, Path]) -> Path:
        """Get path for preprocessed image."""
        return self.processed_dir / Path(original_path).name.replace(".jpg", ".png")

    def _preprocess_images(self) -> None:
        """Preprocess and save all images."""
        logger.info("Preprocessing images...")
        resize_transform = T.Compose(
            [T.Resize(self.resize_size), T.CenterCrop(self.resize_size)]
        )

        for img_path in self.image_paths:
            processed_path = self._get_processed_path(img_path)
            if not processed_path.exists():
                try:
                    with Image.open(img_path) as img:
                        img = img.convert("RGB")
                        img = resize_transform(img)
                        img.save(processed_path, "PNG")
                except Exception as e:
                    logger.error(f"Error preprocessing {img_path}: {str(e)}")
                    raise

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get dataset item.

        Args:
            idx: Index of item

        Returns:
            Tuple of (image, label)
        """
        try:
            img_path = self.image_paths[idx]
            processed_path = self._get_processed_path(img_path)

            if not processed_path.exists():
                raise FileNotFoundError(f"Image not found at {processed_path}")

            img = Image.open(processed_path).convert("RGB")
            img_tensor = self.transform(img)

            return img_tensor, self.labels[idx]

        except Exception as e:
            logger.error(f"Error loading image at index {idx}: {str(e)}")
            raise ValueError(f"Error loading image at index {idx}: {str(e)}")
