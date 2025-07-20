# VQM24 PyTorch Geometric Dataset Processing Pipeline

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch Geometric](https://img.shields.io/badge/PyG-2.5.0-orange)](https://pytorch-geometric.readthedocs.io/en/latest/)
[![Docker](https://img.shields.io/badge/Docker-ready-blue.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Table of Contents

* [Project Description](#project-description)
* [Features](#features)
* [Installation](#installation)
* [Usage](#usage)
* [Configuration](#configuration)
* [Project Structure](#project-structure)
* [Citation](#citation)
* [Acknowledgements](#acknowledgements)

## Project Description

This repository provides a robust and configurable data processing pipeline for the **VQM24 (Virtual Quantum Mechanical Property Prediction)** dataset, designed specifically for use with **PyTorch Geometric (PyG)**. The pipeline handles the entire process from raw data acquisition (downloading from Zenodo) to the creation of ready-to-use `torch_geometric.data.Data` objects, incorporating various pre-filtering, feature engineering, and transformation steps.

## Features

* **Automated Data Acquisition**: Seamlessly downloads the VQM24 dataset from a specified URL (e.g., Zenodo).
* **Chunked Data Processing**: Efficiently processes large datasets by loading and processing data in configurable chunks, optimizing memory usage.
* **Configurable Pre-filtering**:
    * **Atom Count Filters**: Filter molecules based on minimum and maximum atom counts.
    * **Heavy Atom Filters**: Include or exclude molecules based on the presence or absence of specified heavy atoms (e.g., `C`, `N`, `O`, `F`, `Br`, `Cl`, `P`, `S`, `Si`).
* **Molecular Data Conversion**: Converts raw molecular data (SMILES, coordinates, atomic numbers) into RDKit molecule objects.
* **PyTorch Geometric Data Object Creation**: Transforms RDKit molecules into `torch_geometric.data.Data` objects, including atom positions (`pos`) and atomic numbers (`z`).
* **Extensive Property Enrichment**: Adds various molecular properties to the PyG Data objects, including:
    * **Scalar Graph Targets**: `homo_hartree`, `lumo_hartree`, `gap_hartree`, `dipole_moment_debye`, `total_energy_hartree`, `energy_atomization_hartree`
    * **Node-level Features**: (e.g., `atom_types`, `partial_charges`)
    * **Graph-level Vector Properties**: (e.g., `eigenvalues`, `frequencies`, `vibmodes`)
    * **Derived Properties**: Calculates atomization energy based on atomic energies.
* **Vibrational Data Refinement**: Cleans and refines vibrational frequencies and modes, handling invalid entries, near-zero frequencies, and ensuring unique (frequency, vibmode) pairs.
* **Structural Feature Engineering**: Adds configurable atom-level (`x`) and bond-level (`edge_attr`) structural features derived from RDKit molecules, such as atom degree, hybridization, formal charge, number of radical electrons, bond type, and bond direction.
* **PyG Pre-Transformations**: Supports the application of standard PyTorch Geometric `pre_transform` operations defined in the configuration, such as `OneHotDegree`.
* **Robust Error Handling**: Implements a comprehensive custom exception hierarchy to gracefully manage various issues, including configuration errors, data processing errors, RDKit conversion failures, and filtering rejections.
* **Centralized Logging**: Configurable logging system to record processing steps, warnings, and errors, with output to both console and a log file.
* **Modular Design**: Structured into multiple Python modules (`config`, `data_refining`, `data_utils`, `exceptions`, `logging_config`, `mol_conversion`, `molecule_filters`, `mol_structural_features`, `property_enrichment`, `vqm24_dataset`) for clarity and maintainability.

## Installation

This project uses `conda` for environment management.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your_username/your_repository_name.git](https://github.com/your_username/your_repository_name.git)
    cd your_repository_name
    ```
2.  **Create and activate the conda environment:**
    A basic `environment.yml` would typically look like this, including the core dependencies inferred from the project:

    ```yaml
    name: shah_env
    channels:
      - pytorch
      - pyg
      - conda-forge
      - defaults
    dependencies:
      - python=3.11
      - numpy
      - pytorch
      - cpuonly # Or cudatoolkit=X.X if you have a GPU
      - rdkit
      - pyyaml
      - scipy
      - torch-geometric
      - pandas
      - matplotlib
      - tqdm
      - requests
    ```
    Create the environment using the `environment.yml` (if provided and populated) or by manually installing:
    ```bash
    # If you have a populated environment.yml:
    conda env create -f environment.yml
    conda activate shah_env
    ```
    **Alternatively, if `environment.yml` is empty or missing, install manually:**
    ```bash
    conda create -n shah_env python=3.11 numpy pytorch cpuonly rdkit pyyaml scipy torch-geometric pandas matplotlib tqdm requests -c pytorch -c pyg -c conda-forge -c defaults
    conda activate shah_env
    ```
    *(Note: For GPU support, replace `cpuonly` with `cudatoolkit=X.X` where `X.X` matches your CUDA version, and ensure you install the correct PyTorch version for CUDA.)*

## Usage

To process the VQM24 dataset and generate the PyTorch Geometric dataset, run the `main.py` script:

```bash
python main.py

The script will:

    Initialize logging.

    Load configurations from config.yaml.

    Download the DFT_all.npz raw data file into ~/Chem_Data/VQM24_PyG_Dataset/raw/ if it doesn't exist.

    Process the data in chunks, applying filters, converting molecules, enriching properties, and applying PyG transformations.

    Save the final processed dataset as data.pt in ~/Chem_Data/VQM24_PyG_Dataset/processed/.

    Perform a quick integrity test on a sample from the processed dataset.

Detailed logs will be printed to the console and saved to a log file (e.g., main.log) in the same directory as main.py.

Example of loading the processed dataset

Once main.py has run successfully and generated data.pt, you can load the dataset as follows:
Python

import torch
from torch_geometric.data import InMemoryDataset
from pathlib import Path

# Define the dataset root directory where data.pt is saved
dataset_root_dir = Path.home() / "Chem_Data" / "VQM24_PyG_Dataset"

# Instantiate the VQM24Dataset to load the processed data
# You don't need to pass all configs for loading if data.pt already exists
# (force_reload=False is implicit if the file exists and is_processed returns True)
class LoadedVQM24Dataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['DFT_all.npz'] # Name of the raw file

    @property
    def processed_file_names(self):
        return ['data.pt'] # Name of the processed file

    def download(self):
        # No need to download if already processed
        pass

    def process(self):
        # No need to process if already processed
        pass

# Load the dataset
loaded_dataset = LoadedVQM24Dataset(root=str(dataset_root_dir))

print(f"Dataset loaded: {loaded_dataset}")
print(f"Number of graphs: {len(loaded_dataset)}")

# Access a sample graph
sample_graph = loaded_dataset[0]
print(f"First graph: {sample_graph}")
print(f"Atomic numbers (z): {sample_graph.z}")
print(f"Positions (pos): {sample_graph.pos}")
if hasattr(sample_graph, 'y') and sample_graph.y is not None:
    print(f"Targets (y): {sample_graph.y}")
if hasattr(sample_graph, 'x') and sample_graph.x is not None:
    print(f"Atom features (x): {sample_graph.x.shape}")
if hasattr(sample_graph, 'edge_attr') and sample_graph.edge_attr is not None:
    print(f"Bond features (edge_attr): {sample_graph.edge_attr.shape}")

Configuration

The config.yaml file is central to customizing the data processing pipeline. It allows you to define:

    global_constants: Factors like har2ev for energy conversions.

    atomic_energies_hartree: Atomic energies used in atomization energy calculations.

    heavy_atom_symbols_to_z: Mapping for atom filtering.

    data_properties_to_include: Specifies which raw properties from the NPZ file should be extracted and added to the PyG Data object at various levels (scalar graph targets, node features, vector graph properties, variable-length graph properties).

    filter_config: Rules for pre-filtering molecules (e.g., max_atoms, min_atoms, heavy_atom_filter with mode and atoms).

    structural_features: Configures which atom-level and bond-level features to compute and add to pyg_data.x and pyg_data.edge_attr respectively.

    transformations: A list of PyTorch Geometric pre_transform operations to apply to the dataset, with support for passing kwargs to the transform constructors (e.g., OneHotDegree).

Review config.yaml for detailed comments and examples of how to customize these settings.

Project Structure

.
├── config.py                     # Handles loading and access of application configuration from config.yaml.
├── config.yaml                   # Main configuration file for dataset processing, filters, and transforms.
├── data_refining.py              # Functions for cleaning and refining molecular vibrational data (frequencies, vibmodes).
├── data_refining_test.py         # Script to test the `data_refining` module.
├── data_utils.py                 # Utility functions for data validation and safe array access.
├── environment.yml               # Conda environment file listing project dependencies. (Note: user provided an empty file)
├── exceptions.py                 # Defines custom exception classes for robust error handling.
├── logging_config.py             # Configures the application's logging system.
├── main.py                       # Main entry point for dataset processing and testing.
├── mol_conversion.py             # Orchestrates conversion from raw data to RDKit and PyG Data objects.
├── mol_conversion_utils.py       # Helper functions for RDKit and PyG Data object creation.
├── molecule_filters.py           # Implements pre-filtering logic for PyG Data objects based on atom counts and types.
├── mol_structural_features.py    # Functions for adding atom and bond structural features to PyG Data objects.
├── property_enrichment.py        # Adds various molecular properties (targets, features) to PyG Data objects.
├── README.md                     # Project overview and instructions (this file)
└── vqm24_dataset.py              # Defines the main PyTorch Geometric `InMemoryDataset` class for VQM24.

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

We would like to express our gratitude to the developers and maintainers of the following open-source libraries and tools, which are integral to this project:

    Docker: For providing a robust platform for building, shipping, and running our application in a consistent and reproducible environment.

    RDKit: For its powerful cheminformatics functionalities, essential for molecular representation and feature extraction.

    PyTorch & PyTorch Geometric: For providing the robust deep learning framework and graph neural network utilities.

    NumPy: For fundamental numerical operations.

    Tqdm: For excellent progress bars, enhancing user experience during data processing.

    Requests & PyYAML: For handling data downloads and configuration parsing respectively.
