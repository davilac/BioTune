"""Simple baseline comparison: BioTune vs Full Fine-Tuning vs Last Layer Only.

This script compares three fine-tuning approaches on the Flowers102 dataset:
1. ft_full: Train all model parameters (baseline)
2. ft_final: Train only the final classification layer (baseline)
3. BioTune: Evolutionary block selection with adaptive learning rates

Quick test with small population and few generations for validation.
"""

import argparse
import csv
import datetime
import logging
import os
import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from src.data.flower102_dataloader import create_dataloaders
from src.models.model_utils import generate_model
from src.optimization.biotuner import BioTuner, OptimizationConfig
from src.optimization.biotuner_problem import FineTuneProblem
from src.training.trainer import ModelTrainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def set_random_seeds(seed: int) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def run_baseline(
    method: str,
    network: str,
    learning_rate: float,
    train_loaders: list,
    val_loaders: list,
    test_loader,
    num_classes: int,
    device: torch.device,
    seeds: list,
    n_epochs: int = 30,
    patience: int = 3,
) -> dict:
    """Run a baseline fine-tuning method.
    
    Args:
        method: Fine-tuning method ('ft_full' or 'ft_final')
        network: Network architecture
        learning_rate: Learning rate
        train_loaders: Training data loaders
        val_loaders: Validation data loaders
        test_loader: Test data loader
        num_classes: Number of classes
        device: Device to use
        seeds: Random seeds for multiple runs
        n_epochs: Number of training epochs
        patience: Early stopping patience
        
    Returns:
        Dictionary with results
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"Running Baseline: {method}")
    logger.info(f"{'='*70}")
    
    val_acc_list = []
    test_acc_list = []
    times_list = []
    
    for i, seed in enumerate(seeds):
        logger.info(f"\nRun {i+1}/{len(seeds)} with seed {seed}")
        set_random_seeds(seed)
        
        # Use first fold for baselines
        fold_id = 0
        
        # Generate model
        model, optimizer, scheduler = generate_model(
            seed=seed,
            network=network,
            ft_method=method,
            learning_rate=learning_rate,
            trainable_set_ids=None,
            lr_ratios=None,
            output_classes=num_classes,
            device=str(device),
        )
        
        # Create trainer
        trainer = ModelTrainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=torch.nn.CrossEntropyLoss(),
            device=device,
            num_classes=num_classes,
            save_dir=Path("models/baseline"),
            network_type=network,
        )
        
        # Train
        start_time = time.time()
        test_acc, best_val_acc, best_epoch, _, _, metrics = trainer.train(
            train_loader=train_loaders[fold_id],
            val_loader=val_loaders[fold_id],
            test_loader=test_loader,
            num_epochs=n_epochs,
            early_stop_patience=patience,
            save_weight_grads=False,
        )
        elapsed_time = time.time() - start_time
        
        val_acc_list.append(best_val_acc)
        test_acc_list.append(test_acc)
        times_list.append(elapsed_time)
        
        logger.info(f"Val Acc: {best_val_acc:.4f}, Test Acc: {test_acc:.4f}, Time: {elapsed_time:.1f}s")
        
        # Clean up
        del model, optimizer, scheduler
        torch.cuda.empty_cache()
    
    results = {
        "method": method,
        "val_acc_mean": np.mean(val_acc_list),
        "val_acc_std": np.std(val_acc_list),
        "test_acc_mean": np.mean(test_acc_list),
        "test_acc_std": np.std(test_acc_list),
        "time_mean": np.mean(times_list),
        "time_std": np.std(times_list),
    }
    
    logger.info(f"\n{method} Results:")
    logger.info(f"Val Accuracy: {results['val_acc_mean']:.4f} ± {results['val_acc_std']:.4f}")
    logger.info(f"Test Accuracy: {results['test_acc_mean']:.4f} ± {results['test_acc_std']:.4f}")
    logger.info(f"Time: {results['time_mean']:.1f}s ± {results['time_std']:.1f}s")
    
    return results


def run_biotune(
    network: str,
    learning_rate: float,
    train_loaders: list,
    val_loaders: list,
    test_loader,
    num_classes: int,
    device: torch.device,
    seeds: list,
    n_epochs: int = 30,
    patience: int = 3,
    n_generations: int = 5,
    population_size: int = 5,
    elite_size: int = 2,
) -> dict:
    """Run BioTune optimization.
    
    Args:
        network: Network architecture
        learning_rate: Learning rate
        train_loaders: Training data loaders
        val_loaders: Validation data loaders
        test_loader: Test data loader
        num_classes: Number of classes
        device: Device to use
        seeds: Random seeds for training
        n_epochs: Number of training epochs per individual
        patience: Early stopping patience
        n_generations: Number of evolutionary generations
        population_size: Size of population
        elite_size: Number of elite individuals
        
    Returns:
        Dictionary with results
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"Running BioTune Optimization")
    logger.info(f"{'='*70}")
    
    # Setup paths
    results_dir = Path("results/baseline_comparison")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = results_dir / f"biotune_log_{timestamp}.csv"
    
    # Initialize log file
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "iteration", "generation", "fold", "seed", "method", "network",
            "max_epochs", "dataset_size", "lr", "genes", "lr_ratios",
            "trainable_set_ids", "best_epoch", "train_loss", "val_loss",
            "test_loss", "train_acc", "val_acc", "test_acc", "test_avg_pres",
            "test_kappa", "test_f1_macro"
        ])
    
    # Setup fitness parameters
    fitness_params = {
        "method": "adaptive_block_normexp",
        "network": network,
        "loss_func": torch.nn.CrossEntropyLoss(),
        "train_loaders": train_loaders,
        "val_loaders": val_loaders,
        "lr": learning_rate,
        "n_epochs": n_epochs,
        "patience": patience,
        "lr_ratios": None,
        "print_level": 2,
        "save_model_path": Path("models/biotune"),
        "device": device,
        "set_size": 6 if network == "resnet50" else 9,  # ResNet50: 6 blocks, DenseNet121: 9 blocks
        "generation_id": 0,
        "log_file_path": log_file,
        "train_split_pct": 0.5,
        "n_classes": num_classes,
        "fitness_var": "acc",
        "use_table": False,
        "save_weight_grads": False,
        "seeds": seeds,
    }
    
    # Create problem instance
    problem = FineTuneProblem(params=fitness_params)
    
    # Setup optimization config
    n_genes = fitness_params["set_size"] + 1  # +1 for threshold gene
    opt_config = OptimizationConfig(
        bounds=np.array([[0, 1]] * n_genes),
        n_generations=n_generations,
        population_size=population_size,
        elite_size=elite_size,
        save_dir=results_dir,
        filename_prefix=f"biotune_{network}",
        device=str(device),
    )
    
    # Create BioTuner
    biotuner = BioTuner(
        config=opt_config,
        fitness_function=problem.compute_fitness,
        update_params_function=problem.update_params,
        fitness_params=fitness_params,
    )
    
    # Run optimization
    start_time = time.time()
    best_genes, best_fitness = biotuner.run()
    elapsed_time = time.time() - start_time
    
    # Get selected blocks
    selected_blocks = np.where(best_genes > best_genes[-1], 1, 0)[:-1]
    selected_block_ids = np.where(selected_blocks == 1)[0].tolist()
    
    # Compute final learning rate ratios
    problem = FineTuneProblem(params=fitness_params)
    lr_ratios = problem._compute_lr_ratios(
        best_genes, selected_blocks.tolist(), "adaptive_block_normexp"
    )
    
    logger.info(f"\n{'='*70}")
    logger.info(f"BioTune Optimization Complete!")
    logger.info(f"{'='*70}")
    logger.info(f"Best Val Accuracy: {1 - best_fitness:.4f}")
    logger.info(f"Selected Blocks: {selected_block_ids}")
    logger.info(f"Total Time: {elapsed_time:.1f}s")
    
    # Now evaluate on test set using the best configuration
    logger.info(f"\nEvaluating best configuration on test set...")
    
    test_acc_list = []
    for i, seed in enumerate(seeds):
        logger.info(f"\nTest run {i+1}/{len(seeds)} with seed {seed}")
        
        # Generate model with best configuration
        model, optimizer, scheduler = generate_model(
            seed=seed,
            network=network,
            ft_method="adaptive_block_normexp",
            learning_rate=learning_rate,
            trainable_set_ids=selected_block_ids,
            lr_ratios=lr_ratios,
            output_classes=num_classes,
            device=str(device),
        )
        
        # Create trainer
        trainer = ModelTrainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=torch.nn.CrossEntropyLoss(),
            device=device,
            num_classes=num_classes,
            save_dir=Path("models/biotune_test"),
            network_type=network,
        )
        
        # Train on full training set (BioTune test evaluation uses 100% data)
        # Create full data loaders for final test evaluation
        from src.data.flower102_dataloader import create_dataloaders as create_dl
        train_full, val_full, _ = create_dl(
            train_split_pct=1.0,  # Use 100% for test evaluation
            seeds=[seed],
            data_dir=Path("flowers"),
            download=False,
            force_preprocess=False,
        )
        
        test_acc, _, _, _, _, _ = trainer.train(
            train_loader=train_full[0],  # Full training data
            val_loader=val_full[0],
            test_loader=test_loader,
            num_epochs=n_epochs,
            early_stop_patience=patience,
            save_weight_grads=False,
        )
        
        test_acc_list.append(test_acc)
        logger.info(f"Test accuracy: {test_acc:.4f}")
        
        # Clean up
        del model, optimizer, scheduler
        torch.cuda.empty_cache()
    
    avg_test_acc = np.mean(test_acc_list)
    std_test_acc = np.std(test_acc_list)
    
    results = {
        "method": "BioTune",
        "best_fitness": best_fitness,
        "best_val_acc": 1 - best_fitness,
        "test_acc_mean": avg_test_acc,
        "test_acc_std": std_test_acc,
        "best_genes": best_genes.tolist(),
        "selected_blocks": selected_block_ids,
        "lr_ratios": {k: float(v) for k, v in lr_ratios.items()},
        "time_total": elapsed_time,
        "n_generations": n_generations,
    }
    
    logger.info(f"\n{'='*70}")
    logger.info(f"Final BioTune Results:")
    logger.info(f"Val Accuracy: {results['best_val_acc']:.4f}")
    logger.info(f"Test Accuracy: {avg_test_acc:.4f} ± {std_test_acc:.4f}")
    logger.info(f"Selected Blocks: {results['selected_blocks']}")
    logger.info(f"Total Time: {results['time_total']:.1f}s")
    logger.info(f"{'='*70}")
    
    return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="BioTune baseline comparison")
    parser.add_argument("--network", type=str, default="resnet50", choices=["resnet50", "densenet121"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--n_epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--train_split", type=float, default=0.5)
    parser.add_argument("--n_generations", type=int, default=5, help="BioTune generations")
    parser.add_argument("--population_size", type=int, default=5, help="BioTune population size")
    parser.add_argument("--elite_size", type=int, default=2, help="BioTune elite size")
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
    
    # Set seeds for reproducibility
    seeds = [684, 559, 629]  # Use 3 seeds for baseline methods
    set_random_seeds(seeds[0])
    
    # Create data loaders
    logger.info("\nLoading Flowers102 dataset...")
    data_dir = Path("flowers")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Baselines: Use 100% of data (train_split_pct=1.0)
    train_loaders_full, val_loaders_full, test_loader = create_dataloaders(
        train_split_pct=1.0,  # Always 100% for fair comparison
        seeds=seeds,
        data_dir=data_dir,
        download=True,
        force_preprocess=False,
    )
    
    # BioTune: Use folded data for efficient evolutionary search
    train_loaders_folds, val_loaders_folds, _ = create_dataloaders(
        train_split_pct=args.train_split,  # e.g., 0.5 → 2 folds
        seeds=seeds,
        data_dir=data_dir,
        download=False,  # Already downloaded
        force_preprocess=False,
    )
    
    num_classes = 102  # Flowers102
    
    logger.info(f"Baselines will use: 100% of data")
    logger.info(f"BioTune will use: {int(np.ceil(1.0/args.train_split))} folds (train_split_pct={args.train_split})")
    
    # Run experiments
    results_all = []
    
    # 1. Full Fine-Tuning (use 100% of data)
    results_ft_full = run_baseline(
        method="ft_full",
        network=args.network,
        learning_rate=args.lr,
        train_loaders=train_loaders_full,  # 100% of data
        val_loaders=val_loaders_full,
        test_loader=test_loader,
        num_classes=num_classes,
        device=device,
        seeds=seeds,
        n_epochs=args.n_epochs,
        patience=args.patience,
    )
    results_all.append(results_ft_full)
    
    # 2. Last Layer Only (use 100% of data)
    results_ft_final = run_baseline(
        method="ft_final",
        network=args.network,
        learning_rate=args.lr,
        train_loaders=train_loaders_full,  # 100% of data
        val_loaders=val_loaders_full,
        test_loader=test_loader,
        num_classes=num_classes,
        device=device,
        seeds=seeds,
        n_epochs=args.n_epochs,
        patience=args.patience,
    )
    results_all.append(results_ft_final)
    
    # 3. BioTune (use folded data for efficient evolution)
    results_biotune = run_biotune(
        network=args.network,
        learning_rate=args.lr,
        train_loaders=train_loaders_folds,  # Folded data
        val_loaders=val_loaders_folds,
        test_loader=test_loader,
        num_classes=num_classes,
        device=device,
        seeds=seeds,
        n_epochs=args.n_epochs,
        patience=args.patience,
        n_generations=args.n_generations,
        population_size=args.population_size,
        elite_size=args.elite_size,
    )
    results_all.append(results_biotune)
    
    # Print summary
    logger.info(f"\n{'='*70}")
    logger.info("COMPARISON SUMMARY")
    logger.info(f"{'='*70}")
    logger.info(f"{'Method':<20} {'Val Acc':<15} {'Test Acc':<15} {'Time (s)':<15}")
    logger.info("-" * 70)
    
    for result in results_all:
        if result["method"] == "BioTune":
            logger.info(
                f"{result['method']:<20} "
                f"{result['best_val_acc']:.4f}          "
                f"{result['test_acc_mean']:.4f}±{result['test_acc_std']:.4f}  "
                f"{result['time_total']:.1f}"
            )
            logger.info(f"  → Selected blocks: {result['selected_blocks']}")
        else:
            logger.info(
                f"{result['method']:<20} "
                f"{result['val_acc_mean']:.4f}±{result['val_acc_std']:.4f}  "
                f"{result['test_acc_mean']:.4f}±{result['test_acc_std']:.4f}  "
                f"{result['time_mean']:.1f}±{result['time_std']:.1f}"
            )
    
    logger.info("="* 70)
    
    # Print BioTune optimized parameters for reproduction
    biotune_result = [r for r in results_all if r["method"] == "BioTune"][0]
    logger.info(f"\n{'='*70}")
    logger.info("BIOTUNE OPTIMIZED PARAMETERS (For Reproduction)")
    logger.info(f"{'='*70}")
    logger.info(f"\nTo reproduce BioTune results without this library:")
    logger.info(f"1. Use {args.network} pretrained on ImageNet")
    logger.info(f"2. Trainable blocks: {biotune_result['selected_blocks']}")
    logger.info(f"   (Freeze all other blocks)")
    logger.info(f"\n3. Learning rates per block:")
    for block_name, lr_ratio in biotune_result['lr_ratios'].items():
        effective_lr = args.lr * lr_ratio
        status = "TRAIN" if lr_ratio > 0 else "FREEZE"
        logger.info(f"   {block_name:<25} lr_ratio={lr_ratio:.4f}  →  lr={effective_lr:.6f}  [{status}]")
    logger.info(f"\n4. Training settings:")
    logger.info(f"   Base learning rate: {args.lr}")
    logger.info(f"   Epochs: {args.n_epochs}")
    logger.info(f"   Optimizer: Adam")
    logger.info(f"   Scheduler: CosineAnnealingLR (T_max=100)")
    logger.info(f"   Loss: CrossEntropyLoss")
    logger.info(f"\n5. Example PyTorch code:")
    logger.info(f"   # Load pretrained model")
    logger.info(f"   model = torchvision.models.{args.network}(weights='IMAGENET1K_V2')")
    logger.info(f"   ")
    logger.info(f"   # Set requires_grad based on selected blocks")
    logger.info(f"   for name, param in model.named_parameters():")
    logger.info(f"       param.requires_grad = name.startswith({biotune_result['selected_blocks']})")
    logger.info(f"   ")
    logger.info(f"   # Create optimizer with per-block learning rates")
    logger.info(f"   param_groups = []")
    logger.info(f"   for block_name in {list(biotune_result['lr_ratios'].keys())}:")
    logger.info(f"       params = [p for n, p in model.named_parameters() if block_name in n]")
    logger.info(f"       param_groups.append({{'params': params, 'lr': base_lr * lr_ratios[block_name]}})")
    logger.info(f"   ")
    logger.info(f"   optimizer = torch.optim.Adam(param_groups)")
    logger.info(f"   # Then train normally...")
    logger.info("="* 70)
    
    logger.info("\nComparison complete!")


if __name__ == "__main__":
    main()
