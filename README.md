# VQM24 Dataset Processing with PyTorch Geometric

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch Geometric](https://img.shields.io/badge/PyG-2.5.0-orange)](https://pytorch-geometric.readthedocs.io/en/latest/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Table of Contents
- [About](#about)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Citation](#citation)
- [License](#license)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)

## About

This project provides a robust and modular pipeline for processing the VQM24 (Virtual Quantum Mechanical) dataset, transforming raw molecular data (SMILES and 3D coordinates) into [PyTorch Geometric (PyG)](https://pytorch-geometric.readthedocs.io/) `Data` objects. It includes functionalities for dynamic property enrichment, comprehensive data filtering, and flexible configuration management, making it suitable for training Graph Neural Networks (GNNs) on quantum chemical properties.

The VQM24 dataset is designed to benchmark the prediction of quantum mechanical properties using machine learning models, offering a diverse set of molecules with calculated properties. This project aims to facilitate the use of this dataset within the PyTorch Geometric ecosystem.

## Features

* **Configurable Data Processing:** Easily manage data loading, processing, and feature generation via `config.yaml`.
* **RDKit Integration:** Converts SMILES and 3D coordinates into RDKit molecular objects, leveraging RDKit's powerful cheminformatics capabilities.
* **PyTorch Geometric Conversion:** Transforms RDKit molecules into `torch_geometric.data.Data` objects, ready for GNN training.
* **Dynamic Property Enrichment:** Adds specified scalar, vector, and variable-length graph-level properties to the PyG `Data` objects.
* **Comprehensive Data Filtering:** Implements pre-filters based on atom count (min/max) and heavy atom composition (include/exclude specific elements).
* **Robust Error Handling:** Utilizes custom exception classes to clearly differentiate between configuration errors, data processing issues, and intentional data rejections.
* **Modular Design:** Separates concerns into distinct modules (config, data utilities, exceptions, logging, molecule conversion, filters, property enrichment) for maintainability and extensibility.
* **Dedicated Logging System:** Provides structured logging to both console and file for easy debugging and monitoring.
* **Quick Dataset Test:** `main.py` includes a utility to perform an integrity check on a random sample from the processed dataset.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone git@github.com:shahram-boshra/vqm24_database_process.git (https://github.com/shahram-boshra/vqm24_database_process.git)
    cd vqm24_database_process
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    # .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    The project relies on `rdkit` which is often best installed via `conda` for full functionality. If you have Anaconda/Miniconda installed:
    ```bash
    conda install -c conda-forge rdkit
    pip install -r requirements.txt
    ```
    If you prefer to use `pip` exclusively (may require specific system dependencies for `rdkit` on some OS):
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the VQM24 dataset:**
    The `VQM24Dataset` class expects the raw `.npz` file (e.g., `DFT_all.npz`) to be available. You will need to download the original VQM24 dataset and place the `DFT_all.npz` file within the `dataset_root_dir` specified in your `main.py` or `config.yaml`. By default, this is set to `~/Chem_Data/VQM24_PyG_Dataset/`.

## Usage

To process the VQM24 dataset and generate the PyTorch Geometric data objects, run the `main.py` script:

```bash
python main.py

Upon successful execution, the processed dataset chunks (e.g., data_chunk_0.pt, pre_transform_data.pt) will be saved in the directory specified by dataset_root_dir in main.py (default: ~/Chem_Data/VQM24_PyG_Dataset/processed). A log file will also be generated in the same directory as main.py.

The main.py script also includes a quick_dataset_test function that verifies the structure, targets, and applied transformations of a random sample from the generated dataset.
Configuration

The behavior of the dataset processing pipeline is controlled by the config.yaml file. This file allows you to define:

    atomic_energies_hartree: Standard atomic energies used for atomization energy calculation.
    heavy_atom_symbols_to_z: Mapping of heavy atom symbols to their atomic numbers for filtering.
    global_constants: Project-wide constants like har2ev (Hartree to eV conversion).
    data_config: Specifies which raw properties to include as y (scalar targets), node_features_to_add, vector_graph_properties_to_include, and variable_len_graph_properties_to_include.
    filter_config: Defines rules for filtering molecules, including max_atoms, min_atoms, and heavy_atom_filter (with mode and atoms options).
    transformations: A list of PyTorch Geometric transforms to apply to the Data objects.

Example config.yaml snippet:
YAML

# Example: Minimal config relevant to the code
atomic_energies_hartree:
  H: -0.50000
  C: -37.84888
  O: -75.06456
  N: -54.58301
  F: -99.78909
  S: -398.07720
  Cl: -459.48232

heavy_atom_symbols_to_z:
  H: 1
  C: 6
  N: 7
  O: 8
  F: 9
  S: 16
  Cl: 17

global_constants:
  har2ev: 27.211386245988 # Hartree to eV conversion factor

data_config:
  scalar_graph_targets_to_include:
    - energy_U0
    - HOMO_energy_eV
    - LUMO_energy_eV
  atomization_energy_key_name: 'atomization_energy_U0_eV'
  node_features_to_add:
    - partial_charge
  vector_graph_properties_to_include: []
  variable_len_graph_properties_to_include: []

filter_config:
  max_atoms: 50
  min_atoms: 3
  heavy_atom_filter:
    enable: True
    mode: 'exclude' # 'include' or 'exclude'
    atoms: ['F', 'Cl', 'Br', 'I'] # e.g., exclude halogens

transformations:
  - name: 'OneHotDegree'
    kwargs:
      max_degree: 5
  - name: 'AddRandomWalkPE' # Example, may require specific PyG version or custom implementation
    kwargs:
      walk_length: 20
      attr_name: 'rwpe'

Project Structure

.
├── config.yaml               # Application configuration file
├── main.py                   # Main entry point for dataset processing and testing
├── vqm24_dataset.py          # PyTorch Geometric InMemoryDataset implementation for VQM24
├── property_enrichment.py    # Module for adding extra properties to PyG Data objects
├── mol_conversion.py         # Handles high-level molecule conversion to PyG Data
├── mol_conversion_utils.py   # Utility functions for RDKit and PyG conversions
├── molecule_filters.py       # Implements pre-processing filters for molecules
├── data_utils.py             # General data utility functions
├── config.py                 # Module for loading and accessing configuration
├── exceptions.py             # Custom exception classes for robust error handling
├── logging_config.py         # Centralized logging configuration
├── requirements.txt          # Python package dependencies
└── README.md                 # Project documentation (this file)

Citation

If you use the VQM24 dataset or this processing pipeline in your research, please consider citing the original VQM24 paper:
Code snippet

@article{li2024vqm24,
  title={VQM24: A New Dataset for Virtual Quantum Mechanical Property Prediction},
  author={Li, Xiaocheng and Guo, Yuzhi and Li, Jiacai and Cai, Jianfeng and Li, Minghao and Peng, Bo and Li, Jie and Yang, Fan and Li, Guangfu and Yang, Zeyi and Li, Jianan and Shao, Bin},
  journal={arXiv preprint arXiv:2402.04631},
  year={2024}
}

Link: arXiv:2402.04631
License

This project is licensed under the MIT License - see the LICENSE file for details.
Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests.
Acknowledgements

    The developers of the VQM24 dataset.
    The PyTorch and PyTorch Geometric development teams.
    The RDKit community.
    All open-source contributors to the libraries used in this project.

<!-- end list -->
