#!/usr/bin/env python3
"""Quick test to verify BioTune installation is working correctly."""

import sys
import os

print("Testing BioTune installation...")
print("=" * 70)

# Test 1: Import core modules
print("\n1. Testing imports...")
try:
    from src.models.model_utils import generate_model
    print("   ✓ Model utils")
except ImportError as e:
    print(f"   ✗ Model utils: {e}")
    sys.exit(1)

try:
    from src.optimization.biotuner import BioTuner, OptimizationConfig
    print("   ✓ BioTuner")
except ImportError as e:
    print(f"   ✗ BioTuner: {e}")
    sys.exit(1)

try:
    from src.optimization.biotuner_problem import FineTuneProblem
    print("   ✓ FineTuneProblem")
except ImportError as e:
    print(f"   ✗ FineTuneProblem: {e}")
    sys.exit(1)

try:
    from src.data.flower102_dataloader import create_dataloaders
    print("   ✓ Data loaders")
except ImportError as e:
    print(f"   ✗ Data loaders: {e}")
    sys.exit(1)

try:
    from src.training.trainer import ModelTrainer
    print("   ✓ Trainer")
except ImportError as e:
    print(f"   ✗ Trainer: {e}")
    sys.exit(1)

# Test 2: Check PyTorch and CUDA
print("\n2. Testing PyTorch...")
import torch
print(f"   ✓ PyTorch version: {torch.__version__}")
print(f"   ✓ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   ✓ CUDA version: {torch.version.cuda}")
    print(f"   ✓ GPU device: {torch.cuda.get_device_name(0)}")

# Test 3: Test model loading
print("\n3. Testing model generation...")
try:
    import numpy as np
    model, optimizer, scheduler = generate_model(
        seed=42,
        network="resnet50",
        ft_method="ft_full",
        learning_rate=0.001,
        output_classes=102,
        device="cpu",  # Use CPU for testing
    )
    print("   ✓ ResNet50 model created successfully")
    print(f"   ✓ Model has {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"   ✓ Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters")
except Exception as e:
    print(f"   ✗ Model generation failed: {e}")
    sys.exit(1)

# Test 4: Test BioTuner initialization
print("\n4. Testing BioTuner initialization...")
try:
    import numpy as np
    
    # Create dummy fitness params
    fitness_params = {
        "method": "adaptive_block_normexp",
        "network": "resnet50",
        "loss_func": torch.nn.CrossEntropyLoss(),
        "train_loaders": [],  # Empty for testing
        "val_loaders": [],
        "lr": 0.001,
        "n_epochs": 1,
        "patience": 1,
        "print_level": 2,
        "save_model_path": "/tmp",
        "device": "cpu",
        "set_size": 6,
        "generation_id": 0,
        "log_file_path": "/tmp/test.csv",
        "train_split_pct": 0.5,
        "n_classes": 102,
        "fitness_var": "acc",
        "use_table": False,
        "save_weight_grads": False,
        "seeds": [42],
    }
    
    problem = FineTuneProblem(params=fitness_params)
    
    opt_config = OptimizationConfig(
        bounds=np.array([[0, 1]] * 7),
        n_generations=1,
        population_size=2,
        elite_size=1,
        save_dir="/tmp/biotune_test",
        filename_prefix="test",
        device="cpu",
    )
    
    biotuner = BioTuner(
        config=opt_config,
        fitness_function=lambda x: 0.5,  # Dummy fitness
        update_params_function=lambda x: None,
        fitness_params=fitness_params,
    )
    
    print("   ✓ BioTuner initialized successfully")
    print(f"   ✓ Population size: {opt_config.population_size}")
    print(f"   ✓ Elite size: {opt_config.elite_size}")
except Exception as e:
    print(f"   ✗ BioTuner initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("✓ All tests passed! BioTune is ready to use.")
print("\nNext steps:")
print("  1. Quick test (2-3 min):  python example/baseline_comparison.py --n_generations 2 --population_size 3")
print("  2. Full comparison (30-60 min):  python example/baseline_comparison.py")
print("  3. Full experiment (1-2 hrs):  python example/train_model.py")
print("\nFor detailed documentation, see README.md")
print("=" * 70)
