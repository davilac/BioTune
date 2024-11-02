"""Main script for running neural network fine-tuning experiments."""

import argparse
import datetime
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from src.data.flower102_dataloader import create_dataloaders
from src.optimization.biotuner import BioTuner, OptimizationConfig
from src.optimization.biotuner_problem import FineTuneProblem
from src.training.utils import get_device, print_model_info


def setup_logging(output_dir: Path) -> Tuple[Path, Path]:
    """Setup logging configuration.

    Args:
        output_dir: Directory for output files

    Returns:
        Tuple of paths to log file and time summary file
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    log_file = output_dir / f"exp_all_{timestamp}.csv"
    time_summary_file = output_dir / f"time_summary_{timestamp}.csv"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(output_dir / f"experiment_{timestamp}.log"),
            logging.StreamHandler(),
        ],
    )

    return log_file, time_summary_file


def set_random_seeds(seed: int) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed to use
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def get_experiment_config() -> Dict:
    """Get experiment configuration.

    Returns:
        Dictionary of configuration parameters
    """
    return {
        "methods": [
            "adaptive_block_exponential",
        ],
        "networks": [
            "resnet50",
        ],
        "learning_rates": [0.001],
        "train_split_percentages": [0.1],
        "population_sizes": [(3, 1)],  # (population_size, elite_size)
        "n_generations": 4,
        # "seeds": [684, 559, 629],
        "seeds": [684, 559],
    }


def setup_directories() -> Tuple[Path, Path]:
    """Setup experiment directories.

    Returns:
        Tuple of paths to model and results directories
    """
    base_dir = Path.cwd()
    model_dir = base_dir / "models/experiment"
    results_dir = base_dir / "results/experiment"

    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    return model_dir, results_dir


def create_fitness_params(
    method: str,
    network: str,
    learning_rate: float,
    train_loaders: List[DataLoader],
    val_loaders: List[DataLoader],
    model_dir: Path,
    log_file: Path,
    device: torch.device,
    num_classes: int,
    train_split_pct: float,
    seeds: List[int],  # Add seeds parameter
) -> Dict:
    """Create parameters for fitness function.

    Args:
        method: Fine-tuning method
        network: Network architecture
        learning_rate: Learning rate
        train_loaders: Training data loaders
        val_loaders: Validation data loaders
        model_dir: Directory for model checkpoints
        log_file: Path to log file
        device: Device to use
        num_classes: Number of classes
        train_split_pct: Training split percentage
        seeds: List of random seeds for training

    Returns:
        Dictionary of fitness parameters
    """
    # Define set size based on network architecture
    network_set_sizes = {
        "resnet50": 6,
        "densenet121": 9,
        "vgg19": 6,
        "inception_v3": 14,
    }

    if network not in network_set_sizes:
        raise ValueError(f"Unsupported network: {network}")

    return {
        "method": method,
        "network": network,
        "loss_func": torch.nn.CrossEntropyLoss(),
        "train_loaders": train_loaders,
        "val_loaders": val_loaders,
        "lr": learning_rate,
        "n_epochs": 30,
        "patience": 3,
        "lr_ratios": None,
        "print_level": 2,
        "save_model_path": model_dir,
        "device": device,
        "set_size": network_set_sizes[network],
        "generation_id": 0,
        "log_file_path": log_file,
        "train_split_pct": train_split_pct,
        "n_classes": num_classes,
        "fitness_var": "acc",
        "use_table": False,
        "save_weight_grads": method in ["autoRGN", "adaptive_block_rgn"],
        "seeds": seeds,  # Add seeds to parameters
    }


def run_experiment(
    config: Dict,
    model_dir: Path,
    results_dir: Path,
    log_file: Path,
    time_summary_file: Path,
    device: torch.device,
) -> None:
    """Run the main experiment.

    Args:
        config: Experiment configuration
        model_dir: Directory for model checkpoints
        results_dir: Directory for results
        log_file: Path to log file
        time_summary_file: Path to time summary file
        device: Device to use
    """
    logger = logging.getLogger(__name__)

    # Create data directory if it doesn't exist
    data_dir = Path.cwd() / "flowers"
    data_dir.mkdir(parents=True, exist_ok=True)

    for method in config["methods"]:
        for network in config["networks"]:
            for learning_rate in config["learning_rates"]:
                for train_split in config["train_split_percentages"]:
                    logger.info(
                        f"\nStarting experiment with:\n"
                        f"Method: {method}\n"
                        f"Network: {network}\n"
                        f"Learning rate: {learning_rate}\n"
                        f"Train split: {train_split}"
                    )

                    # Create data loaders with preprocessing
                    train_loaders, val_loaders, test_loader = create_dataloaders(
                        train_split_pct=train_split,
                        seeds=config["seeds"],
                        data_dir=data_dir,
                        download=True,
                        force_preprocess=True,
                    )

                    # Create fitness parameters
                    fitness_params = create_fitness_params(
                        method=method,
                        network=network,
                        learning_rate=learning_rate,
                        train_loaders=train_loaders,
                        val_loaders=val_loaders,
                        model_dir=model_dir,
                        log_file=log_file,
                        device=device,
                        num_classes=102,  # Flowers102 dataset
                        train_split_pct=train_split,
                        seeds=config["seeds"],  # Pass seeds from config
                    )

                    # Run optimization for different population sizes
                    for pop_size, elite_size in config["population_sizes"]:
                        # Create experiment name
                        lr_str = f"lr{str(learning_rate).replace('.', '')}"
                        method_str = method.replace("adaptive_block_", "")
                        experiment_name = (
                            f"BT_{network}_{method_str}_{lr_str}_"
                            f"p{pop_size}_e{elite_size}_g{config['n_generations']}"
                        )

                        # Set number of genes based on network architecture
                        n_genes = (
                            fitness_params["set_size"] + 1
                        )  # Add 1 for threshold gene

                        # Create optimization config
                        opt_config = OptimizationConfig(
                            bounds=np.array([[0, 1]] * n_genes),
                            n_generations=config["n_generations"],
                            population_size=pop_size,
                            elite_size=elite_size,
                            save_dir=results_dir / experiment_name,
                            filename_prefix=experiment_name,
                            device=str(device),
                        )

                        # Create problem instance
                        problem = FineTuneProblem(params=fitness_params)

                        # Create and run BioTuner
                        biotuner = BioTuner(
                            config=opt_config,
                            fitness_function=problem.compute_fitness,
                            update_params_function=problem.update_params,
                            fitness_params=fitness_params,
                        )

                        # Record execution time
                        start_time = datetime.datetime.now()
                        best_genes, best_fitness = biotuner.run()
                        execution_time = (
                            datetime.datetime.now() - start_time
                        ).total_seconds()

                        # Log execution time
                        with open(time_summary_file, "a") as f:
                            f.write(f"{experiment_name},{execution_time}\n")

                        logger.info(
                            f"\nExperiment {experiment_name} completed:\n"
                            f"Best fitness: {best_fitness}\n"
                            f"Execution time: {execution_time} seconds"
                        )


def main():
    """Main function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run fine-tuning experiments")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Setup
    device = get_device(args.device)
    set_random_seeds(args.seed)
    model_dir, results_dir = setup_directories()
    log_file, time_summary_file = setup_logging(results_dir)

    # Get experiment configuration
    config = get_experiment_config()

    # Log system information
    logging.info(f"PyTorch version: {torch.__version__}")
    logging.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logging.info(f"CUDA version: {torch.version.cuda}")
        logging.info(f"Using device: {device}")

    # Run experiment
    try:
        run_experiment(
            config=config,
            model_dir=model_dir,
            results_dir=results_dir,
            log_file=log_file,
            time_summary_file=time_summary_file,
            device=device,
        )
    except Exception as e:
        logging.error(f"Experiment failed: {str(e)}", exc_info=True)


if __name__ == "__main__":
    main()
