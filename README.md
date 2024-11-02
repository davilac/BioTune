# BioTune: Evolutionary Optimization for Neural Network Fine-Tuning

BioTuner employs evolutionary algorithms to optimize neural network fine-tuning strategies by evolving layer-specific learning rates and selective training patterns. It combines bio-inspired optimization with adaptive learning techniques to find optimal fine-tuning configurations.

## Project Structure
```
├── src/
│ ├── data/
│ │ ├── base_dataloader.py
│ │ └── flower102_dataloader.py
│ ├── models/
│ │ ├── lora.py
│ │ └── model_utils.py
│ ├── optimization/
│ │ ├── biotuner.py
│ │ └── biotuner_problem.py
│ └── training/
│ ├── trainer.py
│ └── utils.py
├── example/
│ └── train_model.py
└── README.md
```
## Features
-  Bio-inspired optimization for neural network fine-tuning
- Adaptive learning rate adjustment
- Support for multiple network architectures (ResNet50, DenseNet121, VGG19, Inception_v3)
- Benchmark with multiple fine-tuning strategies (full, partial, LoRA)
- Easy-to-use interface for custom datasets
- Comprehensive logging and visualization

## Gene Structure and Encoding

### Basic Gene Structure
  - Each individual represents a fine-tuning configuration
  - Genes encode layer-specific learning rates
  - Last gene serves as an adaptive threshold for layer selection
  - Binary encoding for layer selection (active/inactive)

### Basic Gene Structure
```python
# Gene structure for different architectures
gene_structures = {
    "resnet50": {
        "n_genes": 7,  # 6 blocks + threshold
        "blocks": ["conv1", "layer1", "layer2", "layer3", "layer4", "fc"]
    },
    "densenet121": {
        "n_genes": 10,  # 9 blocks + threshold
        "blocks": ["conv0", "denseblock1", "transition1", "denseblock2", 
                  "transition2", "denseblock3", "transition3", "denseblock4", "classifier"]
    },
    "vgg19": {
        "n_genes": 7,  # 6 blocks + threshold
        "blocks": ["features.0-4", "features.5-8", "features.9-12", 
                  "features.13-16", "features.17-20", "classifier"]
    },
    "inception_v3": {
        "n_genes": 15,  # 14 blocks + threshold
        "blocks": ["Conv2d_1a", "Conv2d_2a", "Conv2d_2b", "Conv2d_3b", "Conv2d_4a",
                  "Mixed_5b-d", "Mixed_6a-e", "Mixed_7a-c", "fc"]
    }
}
```

## Requirements and Installation

### Core Dependencies
```txt
# requirements.txt

# Core Libraries
numpy>=1.21.0
torch>=2.0.0
torchvision>=0.15.0
torchmetrics>=1.0.0

# Utilities
tqdm>=4.65.0  # Progress bars
PyYAML>=6.0   # Configuration file handling
joblib>=1.2.0 # Parallel processing
```

## Environment Setup

### 1. Using conda
```bash
# Create new environment
conda create -n biotuner python=3.9
conda activate biotuner

# Install PyTorch with CUDA support
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt
```

### 2. Using virtualenv
```bash
# Create virtual environment
python -m venv biotuner-env
source biotuner-env/bin/activate  # Linux/Mac
biotuner-env\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

## Example usage

See example/train_model.py for a complete example.

## License

This benchmark suite is licensed under the GNU General Public License v3.0 (GPLv3).



