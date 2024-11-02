from dataclasses import dataclass
from typing import Dict, TypedDict


class DatasetConfig(TypedDict):
    """Type definition for dataset configuration parameters."""

    n_classes: int
    has_val_split: bool
    has_test_split: bool
    is_train_split_balanced: bool
    is_test_split_balanced: bool
    x_varname: str
    y_varname: str


@dataclass
class DatasetSpecs:
    """Constants and specifications for supported datasets."""

    # Image statistics (ImageNet values by default)
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    # Default image sizes
    DEFAULT_RESIZE_SIZE = 232
    DEFAULT_CROP_SIZE = 224
    INCEPTION_RESIZE_SIZE = 342
    INCEPTION_CROP_SIZE = 299

    # Default batch sizes
    DEFAULT_TRAIN_BATCH_SIZE = 96
    DEFAULT_VAL_BATCH_SIZE = 64
    DEFAULT_TEST_BATCH_SIZE = 64


class DatasetRegistry:
    """Registry containing configuration for all supported datasets."""

    CONFIGS: Dict[str, DatasetConfig] = {
        "cifar10": {
            "n_classes": 10,
            "has_val_split": False,
            "has_test_split": True,
            "is_train_split_balanced": True,
            "is_test_split_balanced": True,
            "x_varname": "data",
            "y_varname": "targets",
        },
        "flowers102": {
            "n_classes": 102,
            "has_val_split": True,
            "has_test_split": True,
            "is_train_split_balanced": True,
            "is_test_split_balanced": True,
            "x_varname": "images",
            "y_varname": "labels",
        },
    }

    @classmethod
    def get_config(cls, dataset_name: str) -> DatasetConfig:
        """Get configuration for a specific dataset.

        Args:
            dataset_name: Name of the dataset to get configuration for

        Returns:
            Configuration dictionary for the specified dataset

        Raises:
            ValueError: If the dataset name is not found in the registry
        """
        if dataset_name not in cls.CONFIGS:
            raise ValueError(
                f"Dataset '{dataset_name}' not found. "
                f"Available datasets: {list(cls.CONFIGS.keys())}"
            )
        return cls.CONFIGS[dataset_name]

    @classmethod
    def register_dataset(cls, name: str, config: DatasetConfig) -> None:
        """Register a new dataset configuration.

        Args:
            name: Name of the dataset
            config: Configuration dictionary for the dataset

        Raises:
            ValueError: If the dataset name already exists
        """
        if name in cls.CONFIGS:
            raise ValueError(f"Dataset '{name}' is already registered")
        cls.CONFIGS[name] = config

    @classmethod
    def list_datasets(cls) -> list[str]:
        """Get list of all registered datasets.

        Returns:
            List of dataset names
        """
        return list(cls.CONFIGS.keys())


def get_dataset_config(dataset_name: str) -> DatasetConfig:
    """Convenience function to get dataset configuration.

    Args:
        dataset_name: Name of the dataset

    Returns:
        Dataset configuration dictionary
    """
    return DatasetRegistry.get_config(dataset_name)


# Example usage:
if __name__ == "__main__":
    # Get configuration for a dataset
    flowers_config = get_dataset_config("flowers102")
    print(f"Flowers102 config: {flowers_config}")

    # List all available datasets
    available_datasets = DatasetRegistry.list_datasets()
    print(f"Available datasets: {available_datasets}")

    # Register a new dataset
    new_dataset_config: DatasetConfig = {
        "n_classes": 1000,
        "has_val_split": True,
        "has_test_split": True,
        "is_train_split_balanced": True,
        "is_test_split_balanced": True,
        "x_varname": "images",
        "y_varname": "labels",
    }
    DatasetRegistry.register_dataset("imagenet", new_dataset_config)
