"""Bio-inspired optimization framework for neural network fine-tuning."""

import csv
import datetime
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for BioTuner optimization.

    Args:
        bounds: Parameter bounds (min, max) for each gene
        n_generations: Number of generations to run
        population_size: Size of the population
        elite_size: Number of elite individuals to preserve
        save_dir: Directory to save results
        filename_prefix: Prefix for output files
        device: Device to use for computation
    """

    bounds: np.ndarray
    n_generations: int = 50
    population_size: int = 20
    elite_size: int = 4
    save_dir: Path = Path("results")
    filename_prefix: str = "biotuner"
    device: str = "cuda:0"

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.elite_size >= self.population_size:
            raise ValueError("Elite size must be less than population size")
        self.save_dir = Path(self.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)


class Individual:
    """Individual in the population.

    Args:
        bounds: Parameter bounds for genes
        genes: Optional initial genes
    """

    def __init__(self, bounds: np.ndarray, genes: Optional[np.ndarray] = None):
        self.bounds = bounds
        self.genes = self._initialize_genes(genes)
        self.gradients = np.zeros(bounds.shape[0])
        self.extinction_prob = 0.0
        self.fitness = float("inf")
        self.rgn_ratios: Dict[str, float] = {}

    def _initialize_genes(self, genes: Optional[np.ndarray] = None) -> np.ndarray:
        """Initialize genes either randomly or with given values."""
        if genes is not None:
            return self._validate_genes(genes)

        while True:
            genes = np.random.uniform(
                self.bounds[:, 0], self.bounds[:, 1], size=(self.bounds.shape[0],)
            )
            if self._is_valid_genes(genes):
                return genes

    def _validate_genes(self, genes: np.ndarray) -> np.ndarray:
        """Validate and clip genes to bounds."""
        if genes.shape != (self.bounds.shape[0],):
            raise ValueError(
                f"Expected genes shape {self.bounds.shape[0]}, got {genes.shape}"
            )
        return np.clip(genes, self.bounds[:, 0], self.bounds[:, 1])

    def _is_valid_genes(self, genes: np.ndarray) -> bool:
        """Check if genes represent a valid solution."""
        set_selector = np.where(genes > genes[-1], 1, 0)[:-1]
        return np.sum(set_selector) > 0


class BioTuner:
    """Bio-inspired optimization algorithm for neural network fine-tuning.

    Args:
        config: Optimization configuration
        fitness_function: Function to evaluate individual fitness
        update_params_function: Function to update optimization parameters
        fitness_params: Parameters for fitness function
    """

    def __init__(
        self,
        config: OptimizationConfig,
        fitness_function: Callable,
        update_params_function: Callable,
        fitness_params: Dict[str, Any],
    ):
        self.config = config
        self.compute_fitness = fitness_function
        self.update_params = update_params_function
        self.fitness_params = fitness_params

        self.population: List[Individual] = []
        self.best_individual: Optional[Individual] = None
        self.generation = 0
        self.history: Dict[str, List[float]] = {
            "best_fitness": [],
            "avg_fitness": [],
            "diversity": [],
        }

        self._setup_logging()

    def _setup_logging(self) -> None:
        """Setup logging and result files."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Setup generations log file
        self.generations_file = (
            self.config.save_dir
            / f"{self.config.filename_prefix}_generations_{timestamp}.csv"
        )
        with open(self.generations_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "generation",
                    "individual",
                    "fitness",
                    *[f"gene_{i}" for i in range(self.config.bounds.shape[0])],
                    *[f"gradient_{i}" for i in range(self.config.bounds.shape[0])],
                    "extinction_prob",
                    "selected_blocks",
                ]
            )

        # Setup summary log file
        self.summary_file = (
            self.config.save_dir
            / f"{self.config.filename_prefix}_summary_{timestamp}.csv"
        )
        with open(self.summary_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "generation",
                    "best_fitness",
                    "avg_fitness",
                    "diversity",
                    "best_genes",
                    "selected_blocks",
                ]
            )

    def _initialize_population(self) -> None:
        """Initialize the population."""
        logger.info("\nInitializing Population:")
        logger.info("-" * 60)
        logger.info(f"Creating {self.config.population_size} individuals")
        logger.info(f"Each individual has {self.config.bounds.shape[0]} genes")

        if "train_loaders" in self.fitness_params:
            current_fold = self.fitness_params.get("generation_id", 0) % len(
                self.fitness_params["train_loaders"]
            )
            train_size = len(self.fitness_params["train_loaders"][current_fold].dataset)
            val_size = len(self.fitness_params["val_loaders"][current_fold].dataset)
            logger.info(f"Current fold: {current_fold + 1}")
            logger.info(f"Training samples in current fold: {train_size}")
            logger.info(f"Validation samples in current fold: {val_size}")

        logger.info("-" * 60)
        logger.info("Generating initial population...")

        self.population = []
        for i in range(self.config.population_size):
            individual = Individual(self.config.bounds)
            individual.fitness = self.compute_fitness(individual.genes)
            self.population.append(individual)

            set_selector = np.where(individual.genes > individual.genes[-1], 1, 0)[:-1]
            selected_blocks = np.where(set_selector == 1)[0]
            logger.info(
                f"Individual {i+1}: Fitness = {individual.fitness:.4f}, "
                f"Selected blocks = {selected_blocks}"
            )

        self.population.sort(key=lambda x: x.fitness)
        self.best_individual = self._copy_individual(self.population[0])
        logger.info(f"\nInitial population created")
        logger.info(f"Best initial fitness: {self.best_individual.fitness:.4f}")
        logger.info("-" * 60)

    def _copy_individual(self, individual: Individual) -> Individual:
        """Create a deep copy of an individual."""
        new_individual = Individual(self.config.bounds, individual.genes.copy())
        new_individual.gradients = individual.gradients.copy()
        new_individual.extinction_prob = individual.extinction_prob
        new_individual.fitness = individual.fitness
        new_individual.rgn_ratios = individual.rgn_ratios.copy()
        return new_individual

    def _exploit_elite(
        self, individual: Individual, exploration_rate: float = 0.25
    ) -> None:
        """Exploit elite individual through local search.

        Args:
            individual: Individual to exploit
            exploration_rate: Rate of exploration around current solution
        """
        original_genes = individual.genes.copy()
        best_fitness = individual.fitness

        # Select a single gene to explore between genes[:-1]>genes[-1]
        selected_blocks = np.where(original_genes > original_genes[-1], 1, 0)[:-1]
        selected_indices = np.where(selected_blocks == 1)[0]
        if len(selected_indices) == 0:
            return

        gene_idx = np.random.choice(selected_indices)
        logger.info(
            f"Exploring gene {gene_idx} in elite individual: {individual.genes}"
        )

        for i in [gene_idx]:
            delta = exploration_rate * np.random.uniform(-1, 1)

            # Try positive direction
            individual.genes[i] = np.clip(
                original_genes[i] + delta,
                self.config.bounds[i, 0],
                self.config.bounds[i, 1],
            )
            logger.info(f"Exploring gene {i} in positive direction: {individual.genes}")
            fitness_pos = self.compute_fitness(individual.genes)

            # Try negative direction
            individual.genes[i] = np.clip(
                original_genes[i] - delta,
                self.config.bounds[i, 0],
                self.config.bounds[i, 1],
            )
            logger.info(f"Exploring gene {i} in negative direction: {individual.genes}")
            fitness_neg = self.compute_fitness(individual.genes)

            # Keep best result
            if fitness_pos < best_fitness and fitness_pos < fitness_neg:
                individual.genes[i] = original_genes[i] + delta
                individual.fitness = fitness_pos
                best_fitness = fitness_pos
                logger.info(f"Gene {i} updated to {individual.genes[i]}")
            elif fitness_neg < best_fitness and fitness_neg < fitness_pos:
                individual.genes[i] = original_genes[i] - delta
                individual.fitness = fitness_neg
                best_fitness = fitness_neg
                logger.info(f"Gene {i} updated to {individual.genes[i]}")
            else:
                individual.genes[i] = original_genes[i]
                individual.fitness = best_fitness
                logger.info(f"Gene {i} reverted to {individual.genes[i]}")

    def _crossover(
        self, parent1: Individual, parent2: Individual, rate: float
    ) -> Individual:
        """Perform crossover between two parents.

        Args:
            parent1: First parent
            parent2: Second parent
            rate: Crossover rate

        Returns:
            New individual from crossover
        """
        child = Individual(self.config.bounds)

        def linear_interp(genes1, genes2, ratio):
            return ratio * genes1 + (1 - ratio) * genes2

        # Perform crossover
        # mask = np.random.random(len(parent1.genes)) < rate
        # child.genes = np.where(mask, parent1.genes, parent2.genes)
        child.genes = linear_interp(
            parent1.genes, parent2.genes, np.random.uniform(0, 1, len(parent1.genes))
        )

        # Inherit gradients
        child.gradients = parent1.gradients * np.random.random(
            len(parent1.gradients)
        ) + parent2.gradients * np.random.random(len(parent2.gradients))

        return child

    def _mutate(self, individual: Individual, rate: float, factor: float) -> None:
        """Mutate an individual.

        Args:
            individual: Individual to mutate
            rate: Mutation rate
        """
        mutation_mask = np.random.random(len(individual.genes)) < rate
        logger.info(f"Mutation mask: {mutation_mask}")

        if mutation_mask.any():
            mutation = np.random.uniform(-1, 1, size=len(individual.genes)) * factor

            individual.genes[mutation_mask] += mutation[mutation_mask]
            logger.info(f"Mutated genes: {individual.genes}")
            individual.genes = individual._validate_genes(individual.genes)

    def _adoption(
        self,
        individual: Individual,
        parent1: Individual,
        parent2: Individual,
        prototype: Individual,
    ) -> None:
        """Adopt genes from a prototype individual.

        Args:
            individual: Individual to adopt genes
            parent1: First parent
            parent2: Second parent
            prototype: Prototype individual
        """

        def linear_interp(genes1, genes2, ratio):
            return ratio * genes1 + (1 - ratio) * genes2

        # for gene in range(len(individual.genes)):
        average_genes = (parent1.genes + parent2.genes) / 2
        individual.genes += linear_interp(
            np.random.uniform() * (average_genes - individual.genes),
            np.random.uniform() * (prototype.genes - individual.genes),
            np.random.uniform(),
        )

        individual.genes = individual._validate_genes(individual.genes)

    def _select_parents(
        self, mating_pool: List[Individual]
    ) -> Tuple[Individual, Individual, Individual]:
        """Select parents for reproduction using tournament selection.

        Returns:
            Tuple of selected parents
        """

        def linear_dist(n):
            if n <= 0:
                return 0
            dist = np.random.randint(0, n * (n + 3) // 2)

            if dist == 0:
                return 0
            i = -1
            while dist > 0:
                dist -= n - i
                i += 1
            return i

        parent1 = mating_pool[linear_dist(len(mating_pool))]
        parent2 = mating_pool[linear_dist(len(mating_pool))]
        while parent2 is parent1:
            parent2 = mating_pool[linear_dist(len(mating_pool))]

        prototype = mating_pool[linear_dist(len(mating_pool))]

        return parent1, parent2, prototype

    def _update_extinction_probs(self) -> None:
        """Update extinction probabilities for the population."""
        fitnesses = np.array([ind.fitness for ind in self.population])
        min_fitness = np.min(fitnesses)
        max_fitness = np.max(fitnesses)

        if max_fitness > min_fitness:
            for i, individual in enumerate(self.population):
                individual.extinction_prob = (
                    individual.fitness
                    + min_fitness * ((i / (len(self.population) - 1)) - 1)
                ) / max_fitness

    def _log_generation(self) -> None:
        """Log current generation statistics."""
        # Log individual details
        with open(self.generations_file, "a", newline="") as f:
            writer = csv.writer(f)
            for i, ind in enumerate(self.population):
                writer.writerow(
                    [
                        self.generation,
                        i,
                        ind.fitness,
                        *ind.genes,
                        *ind.gradients,
                        ind.extinction_prob,
                        np.where(ind.genes > ind.genes[-1], 1, 0)[:-1].tolist(),
                    ]
                )

        # Log generation summary
        fitnesses = [ind.fitness for ind in self.population]
        avg_fitness = np.mean(fitnesses)
        diversity = np.std([ind.genes for ind in self.population])

        with open(self.summary_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    self.generation,
                    self.best_individual.fitness,
                    avg_fitness,
                    diversity,
                    self.best_individual.genes.tolist(),
                    np.where(
                        self.best_individual.genes > self.best_individual.genes[-1],
                        1,
                        0,
                    )[:-1].tolist(),
                ]
            )

    def evolve(self) -> None:
        """Perform one generation of evolution."""
        logger.info(f"\nGeneration {self.generation + 1} evolution:")

        # Exploit elite individuals
        logger.info(f"\nExploiting {self.config.elite_size} elite individuals:")
        for i in range(self.config.elite_size):
            prev_fitness = self.population[i].fitness
            self._exploit_elite(self.population[i])
            logger.info(
                f"Elite {i + 1}: {prev_fitness:.4f} -> {self.population[i].fitness:.4f}"
            )

        # Create new population
        new_population = self.population[: self.config.elite_size]
        logger.info("\nGenerating new individuals:")

        # Create a mating pool
        mating_pool = self.population.copy()

        while len(new_population) < self.config.population_size:
            if len(mating_pool) > 2:
                # Select parents
                parent1, parent2, prototype = self._select_parents(mating_pool)
                logger.info(
                    f"\nParents selected: {parent1.fitness:.4f}, {parent2.fitness:.4f}"
                )

                # Create offspring
                crossover_rate = np.random.uniform()
                child = self._crossover(parent1, parent2, rate=crossover_rate)
                logger.info(f"Child after crossover: {child.genes}")
                # Store temporal child
                temporal_child = self._copy_individual(child)

                # Apply mutation
                average_extinction_prob = np.mean(
                    [parent1.extinction_prob, parent2.extinction_prob]
                )
                n_genes = len(child.genes)
                mutation_rate = (1 / n_genes) * (
                    average_extinction_prob * (n_genes - 1) + 1
                )
                self._mutate(child, rate=mutation_rate, factor=average_extinction_prob)
                logger.info(f"Child after mutation: {child.genes}")

                # Apply adoption
                self._adoption(child, parent1, parent2, prototype)
                logger.info(f"Child after adoption: {child.genes}")

                # Update gradients
                child.gradients = child.gradients * np.random.random(n_genes) + (
                    child.genes - temporal_child.genes
                )

                # Evaluate fitness
                child.fitness = self.compute_fitness(child.genes)
                set_selector = np.where(child.genes > child.genes[-1], 1, 0)[:-1]
                selected_blocks = np.where(set_selector == 1)[0]
                logger.info(f"Selected blocks: {selected_blocks}")
                logger.info(f"Child fitness: {child.fitness:.4f}")

                new_population.append(child)

                # Remove parents from mating pool with higher fitness than offsprings
                if parent1.fitness > child.fitness:
                    mating_pool.remove(parent1)
                if parent2.fitness > child.fitness:
                    mating_pool.remove(parent2)
            else:
                logger.info("Mating pool exhausted, generating random child")
                child = Individual(self.config.bounds)
                child.fitness = self.compute_fitness(child.genes)
                new_population.append(child)

        # Update population and sort by fitness
        self.population = new_population
        self.population.sort(key=lambda x: x.fitness)

        # Update best individual if improved
        if self.population[0].fitness < self.best_individual.fitness:
            improvement = self.best_individual.fitness - self.population[0].fitness
            self.best_individual = self._copy_individual(self.population[0])
            logger.info(
                f"\nNew best solution found!"
                f"\nImprovement: {improvement:.4f}"
                f"\nNew best fitness: {self.best_individual.fitness:.4f}"
            )

        # Update extinction probabilities
        self._update_extinction_probs()

        # After evolution, add detailed population summary
        logger.info(f"\n{'-'*60}")
        logger.info("Population Summary:")
        logger.info(
            f"{'Individual':^12} {'Fitness':^12} {'Selected Blocks':^20} {'Extinction Prob':^15}"
        )
        logger.info(f"{'-'*60}")

        for i, ind in enumerate(self.population):
            set_selector = np.where(ind.genes > ind.genes[-1], 1, 0)[:-1]
            selected_blocks = np.where(set_selector == 1)[0]
            logger.info(
                f"{i+1:^12d} {ind.fitness:^12.4f} {str(selected_blocks):^20s} "
                f"{ind.extinction_prob:^15.4f}"
            )

        # Generation statistics
        fitnesses = [ind.fitness for ind in self.population]
        logger.info(f"\nGeneration Statistics:")
        logger.info(f"Best Fitness: {min(fitnesses):.4f}")
        logger.info(f"Worst Fitness: {max(fitnesses):.4f}")
        logger.info(f"Average Fitness: {np.mean(fitnesses):.4f}")
        logger.info(f"Fitness Std Dev: {np.std(fitnesses):.4f}")
        logger.info(
            f"Population Diversity: {np.std([ind.genes for ind in self.population]):.4f}"
        )

        if self.population[0].fitness < self.best_individual.fitness:
            improvement = self.best_individual.fitness - self.population[0].fitness
            self.best_individual = self._copy_individual(self.population[0])
            logger.info(f"\nNew best solution found!")
            logger.info(f"Improvement: {improvement:.4f}")
            logger.info(f"New best fitness: {self.best_individual.fitness:.4f}")

        logger.info(f"{'='*60}")

    def run(self) -> Tuple[np.ndarray, float]:
        """Run the optimization process.

        Returns:
            Tuple of (best_genes, best_fitness)
        """
        logger.info("\n" + "=" * 60)
        logger.info("BioTuner Optimization Configuration:")
        logger.info("=" * 60)
        logger.info(f"Population size: {self.config.population_size}")
        logger.info(f"Elite size: {self.config.elite_size}")
        logger.info(f"Number of generations: {self.config.n_generations}")
        logger.info(f"Number of genes per individual: {self.config.bounds.shape[0]}")
        logger.info(f"Device: {self.config.device}")
        logger.info("-" * 60)
        logger.info("Dataset Information:")
        if "train_loaders" in self.fitness_params:
            train_size = len(self.fitness_params["train_loaders"][0].dataset)
            val_size = len(self.fitness_params["val_loaders"][0].dataset)
            logger.info(f"Training set size: {train_size}")
            logger.info(f"Validation set size: {val_size}")
        logger.info("=" * 60 + "\n")

        self._initialize_population()

        for generation in range(self.config.n_generations):
            self.generation = generation
            self.fitness_params["generation_id"] = generation
            logger.info(
                f"\n{'-'*20} Generation {generation + 1}/{self.config.n_generations} {'-'*20}"
            )

            # Update parameters if needed
            self.update_params(self.fitness_params)

            # Evolve population
            self.evolve()

            # Log results
            self._log_generation()

            # Store and log history
            self.history["best_fitness"].append(self.best_individual.fitness)
            self.history["avg_fitness"].append(
                np.mean([ind.fitness for ind in self.population])
            )
            self.history["diversity"].append(
                np.std([ind.genes for ind in self.population])
            )

            logger.info(
                f"\nGeneration summary:"
                f"\nBest fitness: {self.best_individual.fitness:.4f}"
                f"\nAverage fitness: {self.history['avg_fitness'][-1]:.4f}"
                f"\nPopulation diversity: {self.history['diversity'][-1]:.4f}"
            )

        logger.info("\n" + "=" * 50)
        logger.info("Optimization completed!")
        logger.info(f"Best fitness achieved: {self.best_individual.fitness:.4f}")
        logger.info(f"Best genes found: {self.best_individual.genes}")
        logger.info("=" * 50)

        return self.best_individual.genes, self.best_individual.fitness
