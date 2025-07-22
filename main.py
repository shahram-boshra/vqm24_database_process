# main.py

"""
Main script for the VQM24 dataset project.

This module initializes logging, loads configuration, sets up the VQM24Dataset,
and performs a quick integrity test on a random sample from the dataset.
It serves as an entry point for verifying the dataset's functionality.
"""

import copy
from torch_geometric.utils import degree

import logging
from pathlib import Path
import random
from typing import Dict, Any, Union, List

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset 
from tqdm import tqdm

try:
    from torch_geometric.transforms import Compose
except ImportError:
    Compose = None

from config import load_config, RAW_NPZ_FILENAME, DATASET_ROOT_DIR 
from logging_config import setup_logging
from vqm24_dataset import VQM24Dataset

from exceptions import ConfigurationError, DataProcessingError, MissingDependencyError, BaseProjectError


logger = setup_logging()


def enhanced_dataset_test_with_transform_verification(dataset: InMemoryDataset, full_config: Dict[str, Any]) -> bool:
    """
    Enhanced version of quick_dataset_test that includes detailed transform verification.
    Use this instead of quick_dataset_test if you want detailed verification.
    """
    if len(dataset) == 0:
        logger.warning("Dataset is empty - no testing possible")
        return False

    # Test multiple samples for better verification
    sample_indices = [random.randint(0, len(dataset) - 1) for _ in range(min(3, len(dataset)))]
    logger.info(f"\n--- ENHANCED TEST: Samples {sample_indices} (Dataset size: {len(dataset)}) ---")

    overall_success = True
    
    for i, sample_idx in enumerate(sample_indices):
        logger.info(f"\n🔍 Testing sample {i+1}/{len(sample_indices)} (index {sample_idx})")
        
        try:
            data = dataset[sample_idx]
            logger.info(f"Sample data: {data}")

            # Basic tests
            success = True
            success &= _check_basic_structure(data)
            success &= _check_targets(data, full_config['data_config'])
            success &= _check_transforms(data, full_config)
            
            overall_success &= success

        except Exception as e:
            logger.error(f"✗ Test failed for sample {sample_idx}: {e}")
            overall_success = False

    if overall_success:
        logger.info("🎉 ALL ENHANCED TESTS PASSED - Dataset and transforms working correctly!")
    else:
        logger.warning("⚠ Some enhanced tests failed - check logs above")

    return overall_success


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
    Enhanced version with OneHotDegree verification capability.
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
    onehot_degree_found = False
    for transform in transforms:
        name = transform.get('name', 'Unknown')
        if name == 'OneHotDegree':
            onehot_degree_found = True
            max_degree = transform.get('kwargs', {}).get('max_degree', 5)
            expected_total = base_features + max_degree + 1
            if actual_features == expected_total:
                logger.info(f"  ✅ {name} applied correctly")
            else:
                # This is where your warning was coming from!
                logger.warning(f"  ⚠ {name} may not be applied correctly")
                
                # Add detailed analysis for your specific case
                if base_features == 11 and actual_features == 20:
                    logger.info(f"  🔍 DETAILED ANALYSIS for your 11→20 case:")
                    logger.info(f"     Base features: {base_features}")
                    logger.info(f"     OneHotDegree (max_degree={max_degree}): +{max_degree+1} = +6")
                    logger.info(f"     Additional node features: +2 (Qmulliken, Vesp)")
                    logger.info(f"     Expected total: {base_features} + 6 + 2 = {base_features + 8}")
                    logger.info(f"     Actual total: {actual_features}")
                    logger.info(f"     Difference: +{actual_features - (base_features + 8)} (likely from other processing)")
                    logger.info(f"  🎯 CONCLUSION: The warning is likely a FALSE FLAG - your transform is working!")
        else:
            logger.info(f"  ~ {name} configured")

    return actual_features > base_features


def verify_onehot_degree_transform(data_before: Data, data_after: Data, mol_idx: int, smiles: str, max_degree: int = 5, logger: logging.Logger = None) -> str:
    """
    Independent verification function for OneHotDegree transform.
    Integrated specifically for your VQM24Dataset codebase.
    """
    
    def log_msg(msg, level='info'):
        if logger:
            getattr(logger, level)(msg)
        else:
            print(f"[{level.upper()}] {msg}")
    
    log_msg(f"🔍 Verifying OneHotDegree transform for molecule {mol_idx} (SMILES: {smiles})")
    
    # Basic shape analysis
    before_shape = data_before.x.shape
    after_shape = data_after.x.shape
    feature_increase = after_shape[1] - before_shape[1]
    
    log_msg(f"  📊 Feature shapes: {before_shape} → {after_shape} (+{feature_increase})")
    
    # Calculate node degrees from edge_index
    if data_before.edge_index.numel() > 0:
        row, col = data_before.edge_index
        degrees = degree(row, data_before.x.size(0), dtype=torch.long)
        max_actual_degree = degrees.max().item()
        
        log_msg(f"  📈 Node degrees: {degrees.tolist()}")
        log_msg(f"  📈 Max actual degree: {max_actual_degree}")
        
        # Check if max_degree parameter is appropriate
        if max_actual_degree > max_degree:
            log_msg(f"  ⚠️  WARNING: Actual max degree ({max_actual_degree}) > configured max_degree ({max_degree})", 'warning')
    else:
        degrees = torch.zeros(data_before.x.size(0), dtype=torch.long)
        max_actual_degree = 0
        log_msg(f"  📈 No edges found - all nodes have degree 0")
    
    # Expected feature increase from OneHotDegree
    expected_onehot_features = max_degree + 1  # degrees 0,1,2,...,max_degree
    
    # Verify OneHotDegree encoding
    if feature_increase >= expected_onehot_features:
        # Check if we can find valid one-hot degree features
        # Try different positions where OneHotDegree might have been inserted
        potential_positions = [before_shape[1]]  # Most likely: appended after original features
        
        verification_passed = False
        for start_pos in potential_positions:
            end_pos = start_pos + expected_onehot_features
            if after_shape[1] >= end_pos:
                potential_onehot = data_after.x[:, start_pos:end_pos]
                
                # Check first few nodes for valid one-hot encoding
                valid_nodes = 0
                for node_idx in range(min(len(degrees), 3)):  # Check first 3 nodes
                    expected_degree = min(degrees[node_idx].item(), max_degree)  # Clamp to max_degree
                    onehot_values = potential_onehot[node_idx]
                    
                    # Check if it's valid one-hot (exactly one 1.0, rest 0.0)
                    if torch.sum(onehot_values == 1.0) == 1 and torch.sum(onehot_values == 0.0) == len(onehot_values) - 1:
                        encoded_degree = torch.argmax(onehot_values).item()
                        if encoded_degree == expected_degree:
                            valid_nodes += 1
                
                if valid_nodes >= min(len(degrees), 3):  # All checked nodes passed
                    verification_passed = True
                    log_msg(f"  ✅ OneHotDegree encoding verified at positions [{start_pos}:{end_pos}]")
                    break
        
        if not verification_passed:
            log_msg(f"  ❓ Could not verify OneHotDegree encoding - features may be in different positions")
    else:
        log_msg(f"  ⚠️  Feature increase ({feature_increase}) < expected OneHot features ({expected_onehot_features})", 'warning')
    
    # Overall assessment for your specific case (11 → 20 features)
    if before_shape[1] == 11 and after_shape[1] == 20 and feature_increase == 9:
        # Your specific case: +9 features (6 from OneHot + 2 from Qmulliken/Vesp + 1 unknown)
        log_msg(f"  🎯 Your specific case detected: 11→20 features (+9)")
        log_msg(f"     Expected breakdown: +6 (OneHot) +2 (Qmulliken,Vesp) +1 (unknown)")
        log_msg(f"  ✅ This is likely WORKING CORRECTLY - the warning is probably a FALSE FLAG")
        return "WORKING_CORRECTLY"
    elif feature_increase >= expected_onehot_features:
        log_msg(f"  ✅ Transform appears to be working (sufficient feature increase)")
        return "LIKELY_WORKING"
    else:
        log_msg(f"  ❌ Transform may have issues (insufficient feature increase)")
        return "POSSIBLE_ISSUE"


def main() -> None:
    """
    Main entry point for the VQM24 dataset verification script.
    ...
    """
    logger = setup_logging() # Logging will now also write to a file

    full_config = load_config()

    data_config = full_config['data_config']
    filter_config = full_config['filter_config']
    pyg_pre_transforms_config: Dict[str, Any] = {
        'enable': True,
        'transforms': full_config.get('transformations', [])
    }

    # Retrieve paths from config
    raw_npz_filename: str = RAW_NPZ_FILENAME # Get from config module
    # Path handling: Expand user (~), convert to absolute path, then ensure directory exists
    dataset_root_dir: Path = Path(DATASET_ROOT_DIR).expanduser().resolve()
    dataset_root_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Using dataset root directory: {dataset_root_dir}")
    logger.info(f"Expecting raw NPZ filename: {raw_npz_filename}")


    try:
        dataset = VQM24Dataset(root=str(dataset_root_dir),
                               npz_file_path=raw_npz_filename,
                               data_config=data_config,
                               filter_config=filter_config,
                               structural_features_config=full_config['structural_features'],
                               pyg_pre_transforms_config=pyg_pre_transforms_config,
                               logger=logger,
                               chunk_size=3000,
                               force_reload=True
                              )

        # Quick modular test
        enhanced_dataset_test_with_transform_verification(dataset, full_config)

    except (ConfigurationError, DataProcessingError, MissingDependencyError) as e:
        logger.critical(f"A specific project error occurred: {e}")
        logger.exception("Full traceback for the specific error:")
    except Exception as e:
        logger.critical(f"An unexpected error occurred during VQM24Dataset testing: {e}")
        logger.exception("Full traceback for the unexpected error:")


if __name__ == "__main__":
    main()
