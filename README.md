# VQM24 PyTorch Geometric Dataset Processing Pipeline

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch Geometric](https://img.shields.io/badge/PyG-2.5.0-orange)](https://pytorch-geometric.readthedocs.io/en/latest/)
[![Docker](https://img.shields.io/badge/Docker-ready-blue.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


# **VQM24 PyTorch Geometric Dataset Processing Pipeline**


## **Table of Contents**

* [Project Description](#project-description)
* [Features](#features)
* [Installation](#installation)
* [Usage](#usage)
* [Example of loading the processed dataset](#example-of-loading-the-processed-dataset)
* [Configuration](#configuration)
* [Project Structure](#project-structure)
* [Citation](#citation)
* [Acknowledgements](#acknowledgements)
* [Badges](#badges)

## **Project Description**

This repository provides a robust and configurable data processing pipeline for the **VQM24 (Virtual Quantum Mechanical Property Prediction)** dataset, designed specifically for use with **PyTorch Geometric (PyG)**. The pipeline handles the entire process from raw data acquisition (downloading from Zenodo) to the generation of fully enriched and filtered `torch_geometric.data.Data` objects, ready for graph neural network (GNN) applications.

It leverages RDKit for comprehensive molecular structural feature extraction and integrates various quantum chemical properties. The modular design, extensive configuration options via `config.yaml`, and robust error handling ensure flexibility, reproducibility, and stability in preparing the VQM24 dataset for diverse machine learning tasks.


## **Features**

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

## **Installation**

The recommended way to set up and run this project is by using Docker, which provides a reproducible environment with all necessary Conda dependencies pre-configured.

1.  **Ensure Docker is Installed:**
    Make sure Docker Desktop (for Windows/macOS) or Docker Engine (for Linux) is installed and running on your system. You can download it from [https://www.docker.com/get-started](https://www.docker.com/get-started).

2.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/vqm24-dataset-pipeline.git](https://github.com/your-username/vqm24-dataset-pipeline.git)
    cd vqm24-dataset-pipeline
    ```
    *(Remember to replace `https://github.com/your-username/vqm24-dataset-pipeline.git` with the actual URL of your repository.)*

3.  **Build the Docker Image:**
    This step will create a Docker image named `vqm24-pipeline-env` containing a Conda environment with all project dependencies (RDKit, PyTorch, PyTorch Geometric, etc.) pre-installed as specified in `environment.yml`. This might take some time on the first run as it downloads all packages.

    ```bash
    DOCKER_BUILDKIT=1 docker build --network host -t vqm24-pipeline-env .
    ```
    * `DOCKER_BUILDKIT=1`: Enables the faster BuildKit builder.
    * `--network host`: May be necessary in some network environments (e.g., behind a corporate proxy) to ensure Conda can download packages.
    * `-t vqm24-pipeline-env`: Tags the image with a memorable name.
    * `.`: Specifies the current directory as the build context.

## **Usage**

Once the Docker image is built, you can run the data processing pipeline directly or access an interactive shell within the container. Your local project code and dataset will be accessible inside the container via Docker volumes.

1.  **Configure `config.yaml`:**
    Before running, review and adjust the settings in `config.yaml` to define which features and targets you want to include, set filtering criteria, and specify PyG transforms. **Make these changes on your host machine in the `08` directory.**

2.  **Run the processing script (Recommended for processing):**
    This command will execute `main.py` inside the Docker container within the dedicated `shah_env` Conda environment. Your local `08` directory (containing `main.py` and other modules) will be mounted to `/app/08` in the container, and your `Chem_Data` will be mounted to `/root/Chem_Data`.

    ```bash
    docker run --rm \
      -v "$(pwd)/08:/app/08" \
      -v "$(pwd)/Chem_Data/VQM24_PyG_Dataset:/root/Chem_Data/VQM24_PyG_Dataset" \
      vqm24-pipeline-env conda run -n shah_env python /app/08/main.py
    ```
    * `--rm`: Automatically removes the container after it exits.
    * `-v "$(pwd)/08:/app/08"`: Mounts your local `08` directory (relative to where you run the command, i.e., the project root) to `/app/08` inside the container. This allows `main.py` to be found and for you to modify your code on the host, with changes instantly reflected in the container.
    * `-v "$(pwd)/Chem_Data/VQM24_PyG_Dataset:/root/Chem_Data/VQM24_PyG_Dataset"`: Mounts your local `Chem_Data/VQM24_PyG_Dataset` directory (relative to the project root) to the specified path in the container, making your raw and processed data accessible.
    * `vqm24-pipeline-env`: The name of the Docker image you built.
    * `conda run -n shah_env python /app/08/main.py`: Executes the `main.py` script within the `shah_env` Conda environment.

    This script will:
    * Download the raw `DFT_all.npz` file from Zenodo (if not already present within the mounted `/root/Chem_Data/VQM24_PyG_Dataset` directory).
    * Process the dataset according to `config.yaml` settings.
    * Save the processed `torch_geometric.data.Data` objects to `/root/Chem_Data/VQM24_PyG_Dataset/processed/data.pt`.
    * Perform a quick integrity test on a sample of the processed data.

3.  **Accessing an Interactive Shell (for development/debugging):**
    If you need to explore the environment, debug, or run commands manually within the container, you can launch an interactive shell with the Conda environment activated:

    ```bash
    docker run -it --rm \
      -v "$(pwd)/08:/app/08" \
      -v "$(pwd)/Chem_Data/VQM24_PyG_Dataset:/root/Chem_Data/VQM24_PyG_Dataset" \
      vqm24-pipeline-env bash --login
    ```
    Once inside, you will find the `shah_env` Conda environment automatically activated (indicated by `(shah_env)` in your prompt). You can then `cd /app/08` and run Python scripts or other commands as needed.

## **Example of loading the processed dataset:**

```python
import torch
from torch_geometric.data import DataLoader
from vqm24_dataset import VQM24Dataset
from config import load_config
from logging_config import setup_logging

# Initialize logging and load configuration
logger = setup_logging()
full_config = load_config('/app/08/config.yaml') # IMPORTANT: Adjust path for container environment

# Define dataset parameters
# Path inside the container where raw and processed data will be stored/accessed
root_dir = '/root/Chem_Data/VQM24_PyG_Dataset' 
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

The config.yaml file is central to controlling the data processing pipeline. This file should be edited on your host machine within the 08 directory.

Key sections include:

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
├── Dockerfile                      # Defines the Docker image for the project environment
├── environment.yml                 # Conda environment definition with all dependencies
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
├── logging_config.py               # Configures the application's logging system
├── README.md                       # Project overview and instructions (this file)
└── .gitignore                      # Specifies files/directories to ignore in Git

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

    Requests & PyYAML: For handling data downloads and configuration parsing, respectively.
