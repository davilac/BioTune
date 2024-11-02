"""Implementation of the fine-tuning optimization problem for BioTuner."""

import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from src.models.model_utils import generate_model
from src.training.trainer import ModelTrainer

logger = logging.getLogger(__name__)


class FineTuneProblem:
    """Fine-tuning optimization problem for BioTuner.

    Args:
        params: Dictionary of problem parameters
    """

    def __init__(self, params: Dict):
        self.params = params
        self.iteration = 0
        self.table = {}  # Cache for previously computed fitness values
        self._validate_params()

    def _validate_params(self) -> None:
        """Validate problem parameters."""
        required_params = [
            "method",
            "network",
            "loss_func",
            "train_loaders",
            "val_loaders",
            "lr",
            "n_epochs",
            "patience",
            "print_level",
            "save_model_path",
            "device",
            "set_size",
            "generation_id",
            "log_file_path",
            "train_split_pct",
            "n_classes",
            "fitness_var",
        ]

        missing_params = [
            param for param in required_params if param not in self.params
        ]
        if missing_params:
            raise ValueError(f"Missing required parameters: {missing_params}")

    def update_params(self, params: Dict) -> None:
        """Update problem parameters.

        Args:
            params: New parameter values
        """
        self.params.update(params)
        self._validate_params()

    def _compute_lr_ratios(
        self, genes: np.ndarray, set_selector: List[int], method: str
    ) -> Dict[str, float]:
        """Compute learning rate ratios based on genes and method.

        Args:
            genes: Gene values
            set_selector: Binary list indicating selected sets
            method: Fine-tuning method

        Returns:
            Dictionary of learning rate ratios
        """
        network_configs = {
            "resnet50": {
                "layers": [
                    "conv1",
                    "bn1",
                    "layer1",
                    "layer2",
                    "layer3",
                    "layer4",
                    "fc",
                ],
            },
            "densenet121": {
                "layers": [
                    "features.conv0",
                    "features.norm0",
                    "features.denseblock1",
                    "features.transition1",
                    "features.denseblock2",
                    "features.transition2",
                    "features.denseblock3",
                    "features.transition3",
                    "features.denseblock4",
                    "features.norm5",
                    "classifier",
                ],
            },
            "vgg19": {
                "layers": [
                    "features.0",
                    "features.2",
                    "features.5",
                    "features.7",
                    "features.10",
                    "features.12",
                    "features.14",
                    "features.16",
                    "features.19",
                    "features.21",
                    "features.23",
                    "features.25",
                    "features.28",
                    "features.30",
                    "features.32",
                    "features.34",
                    "classifier",
                ],
            },
            "inception_v3": {
                "layers": [
                    "Conv2d_1a_3x3",
                    "Conv2d_2a_3x3",
                    "Conv2d_2b_3x3",
                    "Conv2d_3b_1x1",
                    "Conv2d_4a_3x3",
                    "Mixed_5b",
                    "Mixed_5c",
                    "Mixed_5d",
                    "Mixed_6a",
                    "Mixed_6b",
                    "Mixed_6c",
                    "Mixed_6d",
                    "Mixed_6e",
                    "AuxLogits",
                    "Mixed_7a",
                    "Mixed_7b",
                    "Mixed_7c",
                    "fc",
                ],
            },
        }

        if self.params["network"] not in network_configs:
            raise ValueError(f"Network {self.params['network']} not supported")

        layers = network_configs[self.params["network"]]["layers"]

        if method == "adaptive_block_discriminative":
            return {
                layer: float(selector) for layer, selector in zip(layers, set_selector)
            }

        elif method == "adaptive_block_scaled":
            max_gene = np.max(genes[:-1])
            return {
                layer: selector * gene / max_gene
                for layer, selector, gene in zip(layers, set_selector, genes[:-1])
            }

        elif method == "adaptive_block_normalized":
            max_gene = np.max(genes[:-1])
            ref_gene = genes[-1]
            return {
                layer: selector * (gene - ref_gene) / (max_gene - ref_gene)
                for layer, selector, gene in zip(layers, set_selector, genes[:-1])
            }

        elif method in ["adaptive_block_exponential", "adaptive_block_rgn"]:
            return {
                layer: selector * np.power(10, 2 * (gene - 0.5))
                for layer, selector, gene in zip(layers, set_selector, genes[:-1])
            }

        else:
            raise ValueError(f"Method {method} not supported")

    def _log_results(
        self,
        generation_id: int,
        fold_id: int,
        seed: int,
        genes: np.ndarray,
        lr_ratios: Dict[str, float],
        selected_set_ids: List[int],
        best_epoch: int,
        metrics: List[float],
    ) -> None:
        """Log training results to CSV file.

        Args:
            generation_id: Current generation ID
            fold_id: Current fold ID
            seed: Random seed used
            genes: Gene values
            lr_ratios: Learning rate ratios
            selected_set_ids: Selected set IDs
            best_epoch: Best epoch number
            metrics: List of metric values
        """
        with open(self.params["log_file_path"], "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    self.iteration,
                    generation_id,
                    fold_id,
                    seed,
                    self.params["method"],
                    self.params["network"],
                    self.params["n_epochs"],
                    self.params["train_split_pct"],
                    self.params["lr"],
                    genes,
                    lr_ratios,
                    selected_set_ids,
                    best_epoch,
                    *metrics,
                ]
            )

    def compute_fitness(self, genes: np.ndarray) -> float:
        """Compute fitness for given genes.

        Args:
            genes: Array of gene values

        Returns:
            Fitness value
        """
        self.iteration += 1
        generation_id = self.params["generation_id"]
        fold_id = generation_id % len(self.params["train_loaders"])

        # Convert genes to layer selection
        set_selector = np.where(genes > genes[-1], 1, 0)[:-1].tolist()
        num_selected_sets = np.sum(set_selector)

        # Log initial information
        logger.info("\n" + "-" * 60)
        logger.info(f"Computing fitness for individual in Generation {generation_id}")
        logger.info(f"Using Fold: {fold_id}")
        logger.info(f"Genome: {[f'{g:.4f}' for g in genes]}")
        logger.info(f"Threshold gene: {genes[-1]:.4f}")

        # Convert selection to set IDs and log selected blocks
        selected_set_ids = [
            set_id
            for set_id, set_selected in enumerate(set_selector)
            if set_selected == 1
        ]
        logger.info(f"Number of selected blocks: {num_selected_sets}")
        logger.info(f"Selected block IDs: {selected_set_ids}")

        # Return worst fitness if no sets selected
        if num_selected_sets == 0:
            logger.info("No blocks selected! Returning worst fitness.")
            metrics = [0] * 9
            self._log_results(
                generation_id, fold_id, 0, genes, {}, selected_set_ids, 0, metrics
            )
            return 1.0 if self.params["fitness_var"] in ["acc", "acc+std"] else 10.0

        # Check cache
        if self.params.get("use_table", False) and str(set_selector) in self.table:
            logger.info("Found in table. Skipping evaluation")
            return self.table[str(set_selector)]

        # Compute and log learning rate ratios
        lr_ratios = self._compute_lr_ratios(genes, set_selector, self.params["method"])
        logger.info("\nLearning rates for selected blocks:")
        for block_id in selected_set_ids:
            layer_name = list(lr_ratios.keys())[block_id]
            lr = lr_ratios[layer_name]
            effective_lr = self.params["lr"] * lr
            logger.info(
                f"Block {block_id} ({layer_name}): "
                f"ratio = {lr:.4f}, "
                f"effective lr = {effective_lr:.6f}"
            )

        # Training loop
        max_val_acc_list = []
        val_loss_list = []
        max_epoch_list = []

        logger.info("\nStarting training with multiple seeds:")
        for i, seed in enumerate(self.params["seeds"]):
            logger.info(
                f"\nTraining run {i+1}/{len(self.params['seeds'])} (seed={seed}):"
            )

            # Generate model, optimizer, and scheduler
            model, optimizer, scheduler = generate_model(
                seed=seed,
                network=self.params["network"],
                ft_method=self.params["method"],
                learning_rate=self.params["lr"],
                trainable_set_ids=selected_set_ids,
                lr_ratios=lr_ratios,
                output_classes=self.params["n_classes"],
                device=self.params["device"],
            )

            # Create trainer
            trainer = ModelTrainer(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                loss_fn=self.params["loss_func"],
                device=self.params["device"],
                num_classes=self.params["n_classes"],
                save_dir=self.params["save_model_path"],
                network_type=self.params["network"],
            )

            # Train model
            _, best_val_acc, best_epoch, _, _, metrics = trainer.train(
                train_loader=self.params["train_loaders"][fold_id],
                val_loader=self.params["val_loaders"][fold_id],
                num_epochs=self.params["n_epochs"],
                early_stop_patience=self.params["patience"],
                save_weight_grads=self.params["method"]
                in ["autoRGN", "adaptive_block_rgn"],
            )

            max_val_acc_list.append(best_val_acc)
            val_loss_list.append(metrics[1])
            max_epoch_list.append(best_epoch)

            logger.info(f"Run {i+1} results:")
            logger.info(f"Best validation accuracy: {best_val_acc:.4f}")
            logger.info(f"Best epoch: {best_epoch}")
            logger.info(f"Validation loss: {metrics[1]:.4f}")

            # Log results
            self._log_results(
                generation_id,
                fold_id,
                seed,
                genes,
                lr_ratios,
                selected_set_ids,
                best_epoch,
                metrics,
            )

            # Clean up
            del model, optimizer, scheduler
            torch.cuda.empty_cache()

        # Compute final fitness
        avg_val_acc = np.mean(max_val_acc_list)
        std_val_acc = np.std(max_val_acc_list)
        avg_val_loss = np.mean(val_loss_list)
        avg_max_epoch = np.mean(max_epoch_list)

        if self.params["fitness_var"] == "acc":
            fitness = 1 - avg_val_acc
        elif self.params["fitness_var"] == "loss":
            fitness = avg_val_loss
        elif self.params["fitness_var"] == "acc+std":
            fitness = 1 - avg_val_acc + std_val_acc
        else:
            raise ValueError(
                f"Unsupported fitness variable: {self.params['fitness_var']}"
            )

        logger.info("\nFinal Results:")
        logger.info(
            f"Average validation accuracy: {avg_val_acc:.4f} ± {std_val_acc:.4f}"
        )
        logger.info(f"Average validation loss: {avg_val_loss:.4f}")
        logger.info(f"Average best epoch: {avg_max_epoch:.2f}")
        logger.info(f"Final fitness: {fitness:.4f}")
        logger.info("-" * 60 + "\n")

        return fitness
