# VQM24 PyTorch Geometric Dataset Processing Pipeline

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch Geometric](https://img.shields.io/badge/PyG-2.5.0-orange)](https://pytorch-geometric.readthedocs.io/en/latest/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


## Project Description

This repository provides a robust and configurable data processing pipeline for the **VQM24 (Virtual Quantum Mechanical Property Prediction)** dataset, designed specifically for use with **PyTorch Geometric (PyG)**. The pipeline handles the entire process from raw data acquisition (downloading from Zenodo) to the generation of fully enriched and filtered `torch_geometric.data.Data` objects, ready for graph neural network (GNN) applications.

It leverages RDKit for comprehensive molecular structural feature extraction and integrates various quantum chemical properties. The modular design, extensive configuration options via `config.yaml`, and robust error handling ensure flexibility, reproducibility, and stability in preparing the VQM24 dataset for diverse machine learning tasks.

## Features

* **Automatic Data Download:** Seamlessly downloads the VQM24 dataset (NPZ file) from Zenodo.
* **Configurable Data Enrichment:**
    * Extracts a wide range of **atom-level structural features** (e.g., degree, hybridization, aromaticity) using RDKit.
    * Extracts **bond-level structural features** (e.g., bond type, conjugation, aromaticity) using RDKit.
    * Integrates **quantum mechanical properties** as graph-level targets (e.g., total energy, HOMO-LUMO gap) and node-level features (e.g., Mulliken charges, electrostatic potential).
    * Supports fixed-size vector properties (e.g., dipole, quadrupole) and variable-length properties (e.g., frequencies, vibrational modes).
    * **Calculates atomization energy** as a derived target property based on specified total energy and atomic energies.
* **Flexible Data Filtering:**
    * Pre-filters molecules based on configurable criteria such as maximum/minimum atom count.
    * Supports inclusion or exclusion of molecules based on the presence of specific heavy atoms.
* **PyTorch Geometric Integration:** Converts processed molecular data into `torch_geometric.data.Data` objects.
* **PyG Pre-Transformation Support:** Applies standard PyTorch Geometric transforms (e.g., `OneHotDegree`, `NormalizeFeatures`) as `pre_transform` before saving the dataset.
* **Chunked Processing:** Efficiently processes large datasets by handling data in configurable chunks, optimizing memory usage during the processing stage.
* **Robust Error Handling:** Implements custom exceptions and graceful handling for various issues during molecule conversion, feature extraction, and filtering, ensuring the pipeline continues processing even with problematic data points.
* **Persistent Storage:** Saves the processed dataset as a single `.pt` file for fast loading in subsequent runs.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone git@github.com:shahram-boshra/vqm24_database_process.git (https://github.com/shahram-boshra/vqm24_database_process.git)
    cd vqm24-dataset-pipeline
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    The project relies on `rdkit`, `torch`, `torch-geometric`, `numpy`, `PyYAML`, and `requests`. Install them using pip:
    ```bash
    pip install torch==2.2.0 torch_geometric==2.5.0 # Adjust versions as needed for your CUDA/system
    pip install rdkit numpy pyyaml requests tqdm
    ```
    *Note: Ensure you install `torch-geometric` compatible with your PyTorch version and CUDA setup.*

## Usage

The primary entry point for processing the dataset is `main.py`.

1.  **Configure `config.yaml`:**
    Before running, review and adjust the settings in `config.yaml` to define which features and targets you want to include, set filtering criteria, and specify PyG transforms.

2.  **Run the processing script:**
    ```bash
    python main.py
    ```
    This script will:
    * Download the raw `DFT_all.npz` file from Zenodo (if not already present).
    * Process the dataset according to `config.yaml` settings.
    * Save the processed `torch_geometric.data.Data` objects to `data/processed/DFT_all.pt`.
    * Perform a quick integrity test on a sample of the processed data.

### Example of loading the processed dataset:

```python
import torch
from torch_geometric.data import DataLoader
from vqm24_dataset import VQM24Dataset
from config import load_config
from logging_config import setup_logging

# Initialize logging and load configuration
logger = setup_logging()
full_config = load_config('config.yaml')

# Define dataset parameters
root_dir = './data' # Directory where raw and processed data will be stored
npz_file_name = 'DFT_all.npz'

# Instantiate the dataset
dataset = VQM24Dataset(
    root=root_dir,
    npz_file_path=npz_file_name,
    data_config=full_config['data_config'],
    filter_config=full_config['filter_config'],
    pyg_pre_transforms_config=full_config['transformations'],
    logger=logger
)

# Access dataset information
print(f"Dataset loaded: {dataset.processed_paths[0]}")
print(f"Number of samples: {len(dataset)}")
print(f"First sample data object: {dataset[0]}")

# Use with DataLoader
loader = DataLoader(dataset, batch_size=32, shuffle=True)
for batch in loader:
    print(f"Batch shape (example): {batch}")
    # Your model training loop here
    break

Configuration

The config.yaml file is central to controlling the data processing pipeline. Key sections include:

    global_constants: Defines conversion factors and lookup tables.

    atomic_energies_hartree: Atomic energies for atomization energy calculation.

    heavy_atom_symbols_to_z: Mapping for heavy atom filtering.

    structural_features: Specifies RDKit-derived atom and bond features to include in pyg_data.x and pyg_data.edge_attr.

    data_config: Controls which scalar graph targets (pyg_data.y), node features, and other graph properties are extracted from the raw data. Also defines atomization energy calculation parameters.

    filter_config: Sets criteria for filtering molecules (e.g., max_atoms, heavy_atom_filter).

    transformations: Lists PyTorch Geometric transforms (e.g., OneHotDegree, NormalizeFeatures) to be applied as pre_transform.

Refer to the comments within config.yaml for detailed explanations of each parameter.

Project Structure

.
├── config.yaml                     # Main configuration file for the data pipeline
├── main.py                         # Entry point for running the dataset processing
├── vqm24_dataset.py                # Implements the torch_geometric.data.InMemoryDataset for VQM24
├── mol_conversion.py               # Orchestrates RDKit -> PyG Data conversion and enrichment
├── mol_conversion_utils.py         # Helper functions for RDKit and basic PyG Data conversion
├── mol_structural_features.py      # Functions for extracting and adding RDKit structural features
├── property_enrichment.py          # Functions for adding QM properties and calculating derived targets
├── molecule_filters.py             # Implements pre-filtering logic for molecules
├── data_utils.py                   # General utility functions for data validation and access
├── exceptions.py                   # Custom exception classes for robust error handling
└── logging_config.py               # Configures the application's logging system

Citation

If you use the VQM24 dataset or this processing pipeline in your research, please consider citing the original VQM24 paper:

@article{li2024vqm24,
  title={VQM24: A New Dataset for Virtual Quantum Mechanical Property Prediction},
  author={Li, Xiaocheng and Guo, Yuzhi and Li, Jiacai and Cai, Jianfeng and Li, Minghao and Peng, Bo and Li, Jie and Yang, Fan and Li, Guangfu and Yang, Zeyi and Li, Jianan and Shao, Bin},
  journal={arXiv preprint arXiv:2402.04631},
  year={2024}
}

Link: arXiv:2402.04631

Acknowledgements

We would like to express our gratitude to the developers and maintainers of the following open-source libraries, which are integral to this project:

    RDKit: For its powerful cheminformatics functionalities, essential for molecular representation and feature extraction.

    PyTorch & PyTorch Geometric: For providing the robust deep learning framework and graph neural network utilities.

    NumPy: For fundamental numerical operations.

    Tqdm: For excellent progress bars, enhancing user experience during data processing.

    Requests & PyYAML: For handling data downloads and configuration parsing, respectively.

