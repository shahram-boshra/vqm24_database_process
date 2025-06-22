# main.py

"""
Main script for the VQM24 dataset project.

This module initializes logging, loads configuration, sets up the VQM24Dataset,
and performs a quick integrity test on a random sample from the dataset.
It serves as an entry point for verifying the dataset's functionality.
"""

import logging
from pathlib import Path
import random
from typing import Dict, Any, Union, List

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset # InMemoryDataset for dataset type
from tqdm import tqdm

try:
    from torch_geometric.transforms import Compose
except ImportError:
    Compose = None

from config import load_config
from logging_config import setup_logging
from vqm24_dataset import VQM24Dataset

from exceptions import ConfigurationError, DataProcessingError, MissingDependencyError, BaseProjectError


logger = setup_logging()


def quick_dataset_test(dataset: InMemoryDataset, full_config: Dict[str, Any]) -> bool:
    """
    Performs a quick modular test on a random sample from the provided dataset.

    This function fetches a random data sample, checks its basic PyG structure,
    verifies target configuration, and inspects applied transformations.
    Logs detailed success/failure messages.

    Args:
        dataset (InMemoryDataset): The PyTorch Geometric InMemoryDataset to test.
        full_config (Dict[str, Any]): The complete configuration dictionary
                                       loaded from `config.yaml`.

    Returns:
        bool: True if all checks pass for the random sample, False otherwise.
              Returns False immediately if the dataset is empty or if
              a `BaseProjectError` or unexpected error occurs during testing.
    """
    if len(dataset) == 0:
        logger.warning("Dataset is empty - no testing possible")
        return False

    # Pick a random sample
    sample_idx = random.randint(0, len(dataset) - 1)
    logger.info(f"\n--- QUICK TEST: Random Sample {sample_idx} (Dataset size: {len(dataset)}) ---")

    try:
        # Get the sample
        data = dataset[sample_idx]
        logger.info(f"Sample data: {data}")

        # Basic structure check
        success = True
        success &= _check_basic_structure(data)
        success &= _check_targets(data, full_config['data_config'])
        success &= _check_transforms(data, full_config)

        if success:
            logger.info("✓ ALL TESTS PASSED - Dataset appears to be working correctly")
        else:
            logger.warning("⚠ Some tests failed - check logs above")

        return success

    except BaseProjectError as e:
        logger.error(f"✗ Test failed due to a project-specific error: {e}")
        logger.debug("Full traceback for project-specific error in quick_dataset_test", exc_info=True)
        return False
    except Exception as e:
        logger.error(f"✗ Test failed with an unexpected error: {e}")
        logger.exception("Full traceback for the unexpected error in quick_dataset_test:")
        return False


def _check_basic_structure(data: Data) -> bool:
    """
    Checks the basic structural integrity of a PyTorch Geometric `Data` object.

    Verifies the presence of essential attributes like 'x' (node features),
    'edge_index', 'pos' (3D positions), and 'z' (atomic numbers),
    and ensures the data object has nodes.

    Args:
        data (Data): The PyTorch Geometric Data object to check.

    Returns:
        bool: True if all basic structural checks pass, False otherwise.
    """
    logger.info("Checking basic structure...")

    checks = [
        (hasattr(data, 'x'), "Node features (x)"),
        (hasattr(data, 'edge_index'), "Edge indices"),
        (hasattr(data, 'pos'), "3D positions"),
        (hasattr(data, 'z'), "Atomic numbers"),
        (data.num_nodes > 0, "Has nodes"),
    ]

    success = True
    for check, description in checks:
        if check:
            logger.info(f"  ✓ {description}")
        else:
            logger.error(f"  ✗ {description}")
            success = False

    # Quick dimension check
    if hasattr(data, 'x') and data.x is not None:
        logger.info(f"  Node features shape: {data.x.shape}")
    if hasattr(data, 'pos'):
        logger.info(f"  Positions shape: {data.pos.shape}")

    return success


def _check_targets(data: Data, data_config: Dict[str, Any]) -> bool:
    """
    Checks if the scalar and other graph-level targets are correctly present
    and shaped in the PyTorch Geometric `Data` object according to the config.

    Verifies the 'y' attribute for scalar targets (including atomization energy)
    and checks for the presence of other specified graph properties.

    Args:
        data (Data): The PyTorch Geometric Data object to check.
        data_config (Dict[str, Any]): The 'data_config' section from the
                                       main configuration.

    Returns:
        bool: True if targets are found and correctly configured, False otherwise.
    """
    logger.info("Checking targets...")

    # Scalar targets
    scalar_targets = data_config.get('scalar_graph_targets_to_include', [])
    if data_config.get('atomization_energy_key_name'):
        scalar_targets = scalar_targets + [data_config['atomization_energy_key_name']]

    if scalar_targets:
        if hasattr(data, 'y') and data.y is not None:
            if data.y.shape[0] == len(scalar_targets):
                logger.info(f"  ✓ Scalar targets: {len(scalar_targets)} targets, shape {data.y.shape}")
                # Show first few values
                for i, name in enumerate(scalar_targets[:3]):
                    logger.info(f"    {name}: {data.y[i].item():.4f}")
                if len(scalar_targets) > 3:
                    logger.info(f"    ... and {len(scalar_targets)-3} more")
            else:
                logger.error(f"  ✗ Target mismatch: {data.y.shape[0]} values vs {len(scalar_targets)} expected")
                return False
        else:
            logger.error(f"  ✗ Missing scalar targets (data.y)")
            return False

    # Quick check for other properties
    other_props = (data_config.get('vector_graph_properties_to_include', []) +
                   data_config.get('variable_len_graph_properties_to_include', []))

    found_props = sum(1 for prop in other_props if hasattr(data, prop))
    if other_props:
        logger.info(f"  ✓ Additional properties: {found_props}/{len(other_props)} found")

    return True


def _check_transforms(data: Data, full_config: Dict[str, Any]) -> bool:
    """
    Checks if PyTorch Geometric transformations appear to have been applied
    correctly to the `Data` object.

    Specifically checks for changes in node feature dimensions, especially
    after transforms like 'OneHotDegree'.

    Args:
        data (Data): The PyTorch Geometric Data object to check.
        full_config (Dict[str, Any]): The complete configuration dictionary.

    Returns:
        bool: True if transforms seem to be applied or no transforms are
              configured; False if an expected effect of a transform is missing.
    """
    logger.info("Checking transforms...")

    transforms = full_config.get('transformations', [])
    if not transforms:
        logger.info("  No transforms configured")
        return True

    if not (hasattr(data, 'x') and data.x is not None):
        logger.warning("  Cannot check transforms - no node features")
        return True

    # Calculate expected dimensions
    data_config = full_config['data_config']
    base_features = 9 + len(data_config.get('node_features_to_add', []))
    actual_features = data.x.shape[1]

    logger.info(f"  Feature dimensions: {base_features} → {actual_features}")

    # Check specific transforms
    for transform in transforms:
        name = transform.get('name', 'Unknown')
        if name == 'OneHotDegree':
            max_degree = transform.get('kwargs', {}).get('max_degree', 5)
            expected_total = base_features + max_degree + 1
            if actual_features == expected_total:
                logger.info(f"  ✓ {name} applied correctly")
            else:
                logger.warning(f"  ⚠ {name} may not be applied correctly")
        else:
            logger.info(f"  ~ {name} configured")

    return actual_features > base_features  # At least some transform effect


# --- Main execution block to test the VQM24Dataset class ---
def main() -> None:
    """
    Main entry point for the VQM24 dataset verification script.

    Initializes logging, loads application configuration, prepares dataset
    parameters, instantiates the VQM24Dataset, and performs a quick
    integrity check on a random sample. Handles and logs various
    project-specific and unexpected exceptions.
    """
    logger = setup_logging() # Logging will now also write to a file

    full_config = load_config()
    data_config = full_config['data_config']
    filter_config = full_config['filter_config']
    pyg_pre_transforms_config: Dict[str, Any] = { # Explicitly type this dictionary
        'enable': True, # Explicitly enable PyG transforms if they are defined in config.yaml
        'transforms': full_config.get('transformations', []) # Get transforms list from 'transformations' key
    }

    raw_npz_filename: str = "DFT_all.npz" # Type hint for clarity
    dataset_root_dir: Path = Path.home() / "Chem_Data" / "VQM24_PyG_Dataset" # Type hint for clarity
    dataset_root_dir.mkdir(parents=True, exist_ok=True)

    try:
        dataset = VQM24Dataset(root=str(dataset_root_dir),
                               npz_file_path=raw_npz_filename,
                               data_config=data_config,
                               filter_config=filter_config,
                               pyg_pre_transforms_config=pyg_pre_transforms_config,
                               logger=logger,
                               chunk_size=3000,
                               force_reload=True # Primarily guarantees that the processed data (data.pt and associated chunk files) will be deleted and re-created
                              )

        # Quick modular test
        quick_dataset_test(dataset, full_config)

    except (ConfigurationError, DataProcessingError, MissingDependencyError) as e:
        logger.critical(f"A specific project error occurred: {e}")
        logger.exception("Full traceback for the specific error:")
    except Exception as e:
        logger.critical(f"An unexpected error occurred during VQM24Dataset testing: {e}")
        logger.exception("Full traceback for the unexpected error:")


if __name__ == "__main__":
    main()
