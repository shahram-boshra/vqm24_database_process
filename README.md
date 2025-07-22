# VQM24 PyTorch Geometric Dataset Processing Pipeline

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch Geometric](https://img.shields.io/badge/PyG-2.5.0-orange)](https://pytorch-geometric.readthedocs.io/en/latest/)
[![Docker](https://img.shields.io/badge/Docker-ready-blue.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository provides a robust and configurable data processing pipeline for the **VQM24 (Virtual Quantum Mechanical Property Prediction)** dataset, specifically designed for use with **PyTorch Geometric (PyG)**. The pipeline handles the entire workflow from raw data acquisition to creating production-ready `torch_geometric.data.Data` objects with comprehensive molecular features and properties.

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Examples](#examples)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Features

### 🚀 **Core Capabilities**
- **Automated Data Acquisition**: Seamlessly downloads VQM24 dataset from Zenodo
- **Memory-Efficient Processing**: Chunked data processing for handling large datasets
- **Flexible Filtering**: Configurable pre-filtering based on atom counts and heavy atom types

### 🔬 **Molecular Data Processing**
- **RDKit Integration**: Converts raw molecular data (SMILES, coordinates, atomic numbers) to RDKit objects
- **PyG Data Creation**: Transforms molecules into `torch_geometric.data.Data` objects
- **Rich Property Enrichment**: Adds comprehensive molecular properties including:
  - Scalar targets: HOMO/LUMO energies, dipole moments, total energies
  - Node features: Atom types, partial charges
  - Graph properties: Eigenvalues, vibrational frequencies and modes
  - Derived properties: Atomization energies

### ⚙️ **Advanced Features**
- **Vibrational Data Refinement**: Cleans frequencies and modes, handles invalid entries
- **Structural Feature Engineering**: Configurable atom and bond-level features
- **PyG Transformations**: Built-in support for standard PyG pre-transforms
- **Robust Error Handling**: Comprehensive exception hierarchy for graceful error management
- **Centralized Logging**: Configurable logging with console and file output

## Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/vqm24-pytorch-geometric.git
cd vqm24-pytorch-geometric

# Create conda environment
conda env create -f environment.yml
conda activate shah_env

# Run the processing pipeline
python main.py
```

## Installation

### Prerequisites
- Python 3.11+
- Conda or Miniconda

### Environment Setup

1. **Clone the repository:**
   ```bash
   git clone git@github.com:shahram-boshra/vqm24_database_process.git
   cd vqm24-pytorch-geometric
   ```

2. **Create conda environment:**
   ```bash
   conda env create -f environment.yml
   conda activate shah_env
   ```

   **Alternative manual installation:**
   ```bash
   conda create -n shah_env python=3.11 \
     numpy pytorch cpuonly rdkit pyyaml scipy \
     torch-geometric pandas matplotlib tqdm requests \
     -c pytorch -c pyg -c conda-forge -c defaults
   conda activate shah_env
   ```

   > **Note:** For GPU support, replace `cpuonly` with `cudatoolkit=X.X` matching your CUDA version.

### Docker Support
```bash
# Build Docker image
docker build -t vqm24-processor .

# Run container
docker run -v $(pwd)/data:/app/data vqm24-processor
```

## Usage

### Basic Usage

Run the complete processing pipeline:

```bash
python main.py
```

The pipeline will:
1. Initialize logging and load configurations
2. Download raw data (`DFT_all.npz`) to `~/Chem_Data/VQM24_PyG_Dataset/raw/`
3. Process data in chunks with filtering and feature engineering
4. Save processed dataset as `data.pt` in `~/Chem_Data/VQM24_PyG_Dataset/processed/`
5. Perform integrity tests on the processed data

### Loading Processed Dataset

```python
import torch
from torch_geometric.data import InMemoryDataset
from pathlib import Path

# Define dataset path
dataset_root = Path.home() / "Chem_Data" / "VQM24_PyG_Dataset"

class VQM24Dataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def raw_file_names(self):
        return ['DFT_all.npz']
    
    @property
    def processed_file_names(self):
        return ['data.pt']

# Load dataset
dataset = VQM24Dataset(root=str(dataset_root))
print(f"Loaded {len(dataset)} molecular graphs")

# Access sample data
sample = dataset[0]
print(f"Atoms: {sample.z.shape[0]}")
print(f"Features: {sample.x.shape if hasattr(sample, 'x') else 'None'}")
```

## Configuration

The `config.yaml` file controls all aspects of the processing pipeline:

### Key Configuration Sections

```yaml
# Global constants and conversions
global_constants:
  har2ev: 27.211386245988

# Atomic energies for atomization calculations
atomic_energies_hartree:
  H: -0.500607632585
  C: -37.8302333826
  # ... more elements

# Data properties to extract
data_properties_to_include:
  scalar_graph_targets:
    - homo_hartree
    - lumo_hartree
    - gap_hartree
  
# Filtering configuration
filter_config:
  max_atoms: 50
  min_atoms: 3
  heavy_atom_filter:
    mode: "include_only"
    atoms: ["C", "N", "O", "F"]

# Structural features
structural_features:
  atom_features:
    - degree
    - hybridization
    - formal_charge
  bond_features:
    - bond_type
    - bond_dir

# PyG transformations
transformations:
  - name: "OneHotDegree"
    kwargs:
      max_degree: 10
```

For detailed configuration options, see the commented `config.yaml` file.

## Project Structure

```
vqm24-pytorch-geometric/
├── config.py                    # Configuration management
├── config.yaml                  # Main configuration file
├── data_refining.py             # Vibrational data cleaning
├── data_utils.py                # Data validation utilities
├── exceptions.py                # Custom exception classes
├── logging_config.py            # Logging configuration
├── main.py                      # Main processing script
├── mol_conversion.py            # Molecule conversion orchestration
├── mol_conversion_utils.py      # RDKit and PyG utilities
├── molecule_filters.py          # Pre-filtering logic
├── mol_structural_features.py   # Structural feature extraction
├── property_enrichment.py       # Property addition to PyG objects
├── vqm24_dataset.py            # Main PyG dataset class
├── environment.yml             # Conda environment
├── Dockerfile                  # Docker configuration
└── README.md                   # This file
```

## Examples

### Custom Filtering Example

```python
# Modify config.yaml for custom filtering
filter_config:
  max_atoms: 30
  min_atoms: 5
  heavy_atom_filter:
    mode: "exclude"
    atoms: ["Br", "I"]  # Exclude bromine and iodine
```

### Adding Custom Features

```python
# In mol_structural_features.py, add custom atom features
def get_custom_atom_feature(atom):
    """Custom feature: atom electronegativity"""
    electronegativity_map = {
        'H': 2.20, 'C': 2.55, 'N': 3.04, 'O': 3.44, 'F': 3.98
    }
    return electronegativity_map.get(atom.GetSymbol(), 0.0)
```

### Batch Processing Example

```python
from torch_geometric.loader import DataLoader

# Create data loader for batch processing
dataset = VQM24Dataset(root="path/to/dataset")
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch in loader:
    # Process batch of molecular graphs
    print(f"Batch size: {batch.num_graphs}")
    print(f"Total atoms: {batch.x.shape[0] if hasattr(batch, 'x') else 'N/A'}")
```

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Fork and clone the repository
git clone https://github.com/yourusername/vqm24-pytorch-geometric.git

# Create development environment
conda env create -f environment.yml
conda activate shah_env

# Install in development mode
pip install -e .

# Run tests
python -m pytest tests/
```

### Code Style
- Follow PEP 8 guidelines
- Use type hints where appropriate
- Add docstrings for all public functions
- Include unit tests for new features

## Citation

If you use this processing pipeline or the VQM24 dataset in your research, please cite:

```bibtex
@article{li2024vqm24,
  title={VQM24: A New Dataset for Virtual Quantum Mechanical Property Prediction},
  author={Li, Xiaocheng and Guo, Yuzhi and Li, Jiacai and Cai, Jianfeng and Li, Minghao and Peng, Bo and Li, Jie and Yang, Fan and Li, Guangfu and Yang, Zeyi and Li, Jianan and Shao, Bin},
  journal={arXiv preprint arXiv:2402.04631},
  year={2024}
}
```

**Paper:** [arXiv:2402.04631](https://arxiv.org/abs/2402.04631)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

We thank the developers and maintainers of the following essential libraries:

- **[PyTorch](https://pytorch.org/)** & **[PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)** - Deep learning and graph neural network frameworks
- **[RDKit](https://www.rdkit.org/)** - Cheminformatics toolkit for molecular processing
- **[NumPy](https://numpy.org/)** & **[SciPy](https://scipy.org/)** - Fundamental scientific computing
- **[Docker](https://www.docker.com/)** - Containerization platform
- **[Tqdm](https://tqdm.github.io/)** - Progress bar utilities

## Support

- 📖 **Documentation**: Check the inline code documentation and configuration comments
- 🐛 **Issues**: Report bugs and request features via [GitHub Issues](https://github.com/shahram-boshra/vqm24-pytorch-geometric/issues)
- 💬 **Discussions**: Join our [GitHub Discussions](https://github.com/shahram-boshra/vqm24-pytorch-geometric/discussions)
- 📧 **Contact**: For research collaborations, contact [a.boshra@gmail.com](mailto:a.boshra@gmail.com)

---

**Made with ❤️ for the molecular machine learning community**
