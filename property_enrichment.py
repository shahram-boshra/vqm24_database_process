# property_enrichment.py

"""
This module provides functions to enrich PyTorch Geometric (PyG) Data objects
with various molecular properties.

It includes functionality for adding scalar graph targets, node-level features,
graph-level vector properties, and variable-length graph properties. It also
supports the calculation of derived properties like atomization energy.
The module handles data validation and uses custom exceptions to signal
issues during property enrichment.
"""

import logging
import numpy as np
import torch
from pathlib import Path
import inspect


from typing import Dict, List, Union, Optional, Any
from torch_geometric.data import Data

from config import HAR2EV, ATOMIC_ENERGIES_HARTREE
from data_utils import _is_value_valid_and_not_nan

from exceptions import PropertyEnrichmentError, ConfigurationError


logger = logging.getLogger(__name__)


def add_scalar_graph_targets(
    pyg_data: Data,
    raw_properties_dict: Dict[str, Union[float, int, np.number, np.ndarray, None]],
    molecule_index: int,
    logger: logging.Logger,
    target_keys: List[str]
) -> None:
    """
    Adds scalar graph-level targets to `pyg_data.y`.

    Each target is fetched from `raw_properties_dict` and validated to ensure it is
    a single numeric scalar (int, float, or 1-element NumPy array).
    The collected scalars are concatenated into a `torch.Tensor` and assigned to `pyg_data.y`.

    Args:
        pyg_data (torch_geometric.data.Data): The PyG Data object for the current molecule.
        raw_properties_dict (dict): A dictionary containing all raw data extracted for the molecule.
        molecule_index (int): The index of the molecule being processed.
        logger (logging.Logger): The logger instance.
        target_keys (list): A list of string keys for the scalar targets to add.

    Raises:
        PropertyEnrichmentError: If a target is missing, invalid, has an unexpected
                                 type/shape, or contains NaN/Inf values.
    """
    if not target_keys:
        return

    collected_targets = []
    inchi = getattr(pyg_data, 'inchi', 'N/A') # Get InChI for better error context

    for key in target_keys:
        try:
            # --- Direct access from raw_properties_dict ---
            value = raw_properties_dict.get(key)

            val_to_add = None

            # Case 1: Value is already a direct scalar (int, float, numpy scalar type)
            if isinstance(value, (int, float, np.number)):
                val_to_add = float(value)

            # Case 2: Value is a NumPy array
            elif isinstance(value, np.ndarray):
                if value.size == 1:
                    # If it's a 1-element NumPy array, extract the scalar
                    val_to_add = float(value.item())
                else:
                    # This is the problematic case: a NumPy array with more than one element
                    # Raise PropertyEnrichmentError
                    raise PropertyEnrichmentError(
                        molecule_index=molecule_index,
                        inchi=inchi,
                        property_name=key, # Add property_name
                        reason=f"Scalar target '{key}' has unexpected array shape {value.shape} (size {value.size}). Expected a single scalar or a 1-element array."
                    )
            # Case 3: Value is some other unexpected type (or None if not found in dict)
            else:
                # This handles cases where key might not be in raw_properties_dict or value is None/unexpected
                # Raise PropertyEnrichmentError
                raise PropertyEnrichmentError(
                    molecule_index=molecule_index,
                    inchi=inchi,
                    property_name=key, # Add property_name
                    reason=f"Scalar target '{key}' has unexpected type {type(value)} or is missing. Expected a numeric scalar or NumPy array."
                )

            # After successfully extracting a potential scalar value, check for NaN/Inf
            if not _is_value_valid_and_not_nan(val_to_add):
                # Raise PropertyEnrichmentError
                raise PropertyEnrichmentError(
                    molecule_index=molecule_index,
                    inchi=inchi,
                    property_name=key, # Add property_name
                    reason=f"Scalar target '{key}' has NaN, Inf, or None value after initial conversion."
                )

            collected_targets.append(val_to_add)

        # Catch PropertyEnrichmentError explicitly, then general Exception
        except PropertyEnrichmentError:
            # Re-raise our specific error to be caught by the caller
            raise
        except Exception as e:
            # Catch any other unexpected errors during fetching or conversion
            # Raise PropertyEnrichmentError
            raise PropertyEnrichmentError(
                molecule_index=molecule_index,
                inchi=inchi,
                property_name=key, # Add property_name
                reason=f"Critical error processing scalar target '{key}'.",
                detail=str(e)
            ) from e


    # Only set pyg_data.y if all scalar targets were successfully collected
    if collected_targets:
        pyg_data.y = torch.tensor(collected_targets, dtype=torch.float)


def add_node_features(
    pyg_data: Data,
    raw_properties_dict: Dict[str, Union[np.ndarray, Any]],
    molecule_index: int,
    logger: logging.Logger,
    feature_keys: List[str]
) -> None:
    """
    Adds specified node-level features to `pyg_data.x`.

    Features are retrieved from `raw_properties_dict` and validated to be 1D NumPy arrays
    with a length matching the number of nodes in the PyG Data object.
    These new features are then concatenated with any existing `pyg_data.x` features.

    Args:
        pyg_data (torch_geometric.data.Data): The PyG Data object for the current molecule.
        raw_properties_dict (dict): A dictionary containing all raw data extracted for the molecule.
        molecule_index (int): The index of the molecule being processed.
        logger (logging.Logger): The logger instance.
        feature_keys (list): A list of string keys for the node features to add.

    Raises:
        PropertyEnrichmentError: If a feature is missing, invalid, has an unexpected
                                 format/shape, or if the number of nodes is zero.
    """
    if not feature_keys:
        return

    inchi = getattr(pyg_data, 'inchi', 'N/A') # Get InChI for better error context

    original_x_num_features = pyg_data.x.shape[1] if hasattr(pyg_data, 'x') and pyg_data.x is not None else 0
    additional_node_features_tensors = []

    # Ensure num_nodes is correctly derived from pyg_data.z now (most reliable source of atom count)
    expected_num_nodes = pyg_data.z.size(0) if hasattr(pyg_data, 'z') and pyg_data.z is not None else \
                         pyg_data.x.size(0) if hasattr(pyg_data, 'x') and pyg_data.x is not None else 0

    if expected_num_nodes == 0:
        raise PropertyEnrichmentError(
            molecule_index=molecule_index,
            inchi=inchi,
            reason="Cannot add node features: Number of nodes (derived from z or x) is 0."
        )

    for key in feature_keys:
        try:
            # --- Direct access from raw_properties_dict ---
            node_values = raw_properties_dict.get(key)

            logger.debug(f"      DEBUG: For molecule {molecule_index}, feature '{key}': Fetched value type: {type(node_values)}, shape: {node_values.shape if isinstance(node_values, np.ndarray) else 'N/A'}")
            logger.debug(f"      DEBUG: For molecule {molecule_index}, feature '{key}': is_value_valid_and_not_nan: {_is_value_valid_and_not_nan(node_values)}")

            if not _is_value_valid_and_not_nan(node_values) or \
               not (isinstance(node_values, np.ndarray) and node_values.ndim == 1 and node_values.shape[0] == expected_num_nodes):
                raise PropertyEnrichmentError(
                    molecule_index=molecule_index,
                    inchi=inchi,
                    property_name=key, # Add property_name
                    reason=f"Missing, invalid, or shape mismatch for node feature '{key}' (expected {expected_num_nodes} got {node_values.shape[0] if isinstance(node_values, np.ndarray) else 'N/A'})."
                )
            else:
                additional_node_features_tensors.append(torch.tensor(node_values, dtype=torch.float).unsqueeze(1))
        # Catch PropertyEnrichmentError explicitly, then general Exception
        except PropertyEnrichmentError:
            raise # Re-raise our specific error
        except Exception as e:
            raise PropertyEnrichmentError(
                molecule_index=molecule_index,
                inchi=inchi,
                property_name=key, # Add property_name
                reason=f"Error fetching node-level feature '{key}'.",
                detail=str(e)
            ) from e


    # Only concatenate if we successfully collected all additional features for this molecule
    if additional_node_features_tensors:
        # If pyg_data.x doesn't exist yet (e.g., from_rdmol didn't set it due to simple molecule or an issue)
        if not hasattr(pyg_data, 'x') or pyg_data.x is None or pyg_data.x.numel() == 0:
            pyg_data.x = torch.cat(additional_node_features_tensors, dim=1)
            logger.debug(f"      Initial pyg_data.x was empty. Set to concatenated node features. New shape: {pyg_data.x.shape}")
        else:
            # Before concatenating, ensure current pyg_data.x has the correct number of nodes
            if pyg_data.x.size(0) != expected_num_nodes:
                raise PropertyEnrichmentError(
                    molecule_index=molecule_index,
                    inchi=inchi,
                    reason=f"Node count mismatch before concatenating new features: pyg_data.x has {pyg_data.x.size(0)} nodes, but expected {expected_num_nodes}."
                )

            # Check if features are already there to avoid duplicates, although unlikely with current logic
            if original_x_num_features + len(additional_node_features_tensors) > pyg_data.x.shape[1]:
                pyg_data.x = torch.cat([pyg_data.x] + additional_node_features_tensors, dim=1)
                logger.debug(f"      Appended additional node features. New pyg_data.x shape: {pyg_data.x.shape}")
            else:
                logger.debug(f"      Node features already appear to be present or no new features to add. pyg_data.x shape remains: {pyg_data.x.shape}")

    # After attempting to add, let's verify the final state of x and num_nodes
    if not hasattr(pyg_data, 'x') or pyg_data.x is None or pyg_data.x.numel() == 0:
        raise PropertyEnrichmentError(
            molecule_index=molecule_index,
            inchi=inchi,
            reason="Final check: pyg_data.x is still missing or empty after attempting to add node features."
        )

    # Ensure num_nodes is always correct after x and z are finalized
    if pyg_data.num_nodes != pyg_data.x.size(0):
        logger.warning(f"      pyg_data.num_nodes ({pyg_data.num_nodes}) updated to match pyg_data.x.size(0) ({pyg_data.x.size(0)}) for index {molecule_index}.")
        pyg_data.num_nodes = pyg_data.x.size(0)


def add_vector_graph_properties(
    pyg_data: Data,
    mol_idx: int,
    raw_properties_dict: Dict[str, Union[np.ndarray, None]],
    prop_keys: List[str],
    inchi: str,
    logger: logging.Logger
) -> None:
    """
    Adds specified graph-level vector properties (e.g., dipole, quadrupole, rots)
    to the PyG Data object as attributes (e.g., `pyg_data.dipole`).

    Expects these properties to be 1D NumPy arrays of a fixed size. Special handling
    is provided for 'rots' (rotational constants), allowing it to be a (2,) array
    which will be padded to (3,) for linear molecules.

    Args:
        pyg_data (torch_geometric.data.Data): The PyG Data object for the current molecule.
        mol_idx (int): The index of the molecule being processed.
        raw_properties_dict (dict): A dictionary containing all raw data extracted for the molecule.
        prop_keys (list): A list of string keys for the vector properties to add.
        inchi (str): The InChI string of the molecule, used for error context.
        logger (logging.Logger): The logger instance.

    Raises:
        PropertyEnrichmentError: If any property is invalid, missing, has an unexpected
                                 shape/dimension, or if the molecule has no nodes/positions.
    """
    # Early exit if there are no nodes or valid positions
    if pyg_data.num_nodes == 0 or not hasattr(pyg_data, 'pos') or pyg_data.pos is None or pyg_data.pos.numel() == 0:
        raise PropertyEnrichmentError(
            molecule_index=mol_idx,
            inchi=inchi,
            reason="No nodes or valid positions found for vector graph properties."
        )

    for prop_key in prop_keys:
        try:
            # --- Direct access from raw_properties_dict ---
            value = raw_properties_dict.get(prop_key)

            if not _is_value_valid_and_not_nan(value):
                raise PropertyEnrichmentError(
                    molecule_index=mol_idx,
                    inchi=inchi,
                    property_name=prop_key, # Add property_name
                    reason=f"Missing, invalid, or NaN vector property '{prop_key}'."
                )

            if not isinstance(value, np.ndarray) or value.ndim != 1:
                raise PropertyEnrichmentError(
                    molecule_index=mol_idx,
                    inchi=inchi,
                    property_name=prop_key, # Add property_name
                    reason=f"Vector property '{prop_key}' is not a 1D array. Actual type: {type(value)}, Actual dims: {getattr(value, 'ndim', 'N/A')}."
                )

            # --- 'rots' PADDING ---
            if prop_key == 'rots':
                if value.shape == (2,):
                    # If rots is (2,), pad it to (3,) with a zero
                    # This accounts for linear molecules where one rotational constant is effectively zero/infinite
                    padded_value = np.pad(value, (0, 1), 'constant', constant_values=0.0)
                    logger.debug(f"      Padded 'rots' for index {mol_idx} (InChI: {inchi}) from {value.shape} to {padded_value.shape}.")
                    value = padded_value
                elif value.shape != (3,):
                    raise PropertyEnrichmentError(
                        molecule_index=mol_idx,
                        inchi=inchi,
                        property_name=prop_key, # Add property_name
                        reason=f"Vector property '{prop_key}' has unexpected shape {value.shape}. Expected (3,) or (2,)."
                    )
            # --- END of 'rots' PADDING block ---

            # Explicit shape validation for other specific vector properties
            elif prop_key == 'dipole' and value.shape != (3,):
                raise PropertyEnrichmentError(
                    molecule_index=mol_idx,
                    inchi=inchi,
                    property_name=prop_key, # Add property_name
                    reason=f"Vector property '{prop_key}' has unexpected shape {value.shape}. Expected (3,)."
                )
            elif prop_key == 'quadrupole' and value.shape != (6,):
                raise PropertyEnrichmentError(
                    molecule_index=mol_idx,
                    inchi=inchi,
                    property_name=prop_key, # Add property_name
                    reason=f"Vector property '{prop_key}' has unexpected shape {value.shape}. Expected (6,)."
                )
            elif prop_key == 'octupole' and value.shape != (10,):
                raise PropertyEnrichmentError(
                    molecule_index=mol_idx,
                    inchi=inchi,
                    property_name=prop_key, # Add property_name
                    reason=f"Vector property '{prop_key}' has unexpected shape {value.shape}. Expected (10,)."
                )
            elif prop_key == 'hexadecapole' and value.shape != (15,):
                raise PropertyEnrichmentError(
                    molecule_index=mol_idx,
                    inchi=inchi,
                    property_name=prop_key, # Add property_name
                    reason=f"Vector property '{prop_key}' has unexpected shape {value.shape}. Expected (15,)."
                )

            setattr(pyg_data, prop_key, torch.tensor(value, dtype=torch.float32))
        # Catch PropertyEnrichmentError explicitly, then general Exception
        except PropertyEnrichmentError:
            raise # Re-raise our specific error
        except Exception as e:
            raise PropertyEnrichmentError(
                molecule_index=mol_idx,
                inchi=inchi,
                property_name=prop_key, # Add property_name
                reason=f"Error processing vector property '{prop_key}'.",
                detail=str(e)
            ) from e


def add_variable_len_graph_properties(
    pyg_data: Data,
    raw_properties_dict: Dict[str, Union[np.ndarray, None]],
    molecule_index: int,
    logger: logging.Logger,
    property_keys: List[str]
) -> None:
    """
    Adds specified graph-level properties that can have a variable number of elements
    (e.g., 'freqs' for frequencies, 'vibmodes' for vibrational modes) to the PyG Data object.

    Special handling for 'vibmodes': It expects a 3D NumPy array (num_modes, num_atoms, 3)
    or a 2D array (num_modes * num_atoms, 3) which will be reshaped. It is then stored
    as a Python list of 2D tensors (one [num_atoms, 3] tensor per mode) to accommodate
    varying numbers of modes per molecule during batching.

    Args:
        pyg_data (torch_geometric.data.Data): The PyG Data object for the current molecule.
        raw_properties_dict (dict): A dictionary containing all raw data extracted for the molecule.
        molecule_index (int): The index of the molecule being processed.
        logger (logging.Logger): The logger instance.
        property_keys (list): A list of string keys for the variable-length properties to add.

    Raises:
        PropertyEnrichmentError: If a property is missing, invalid, or has an unexpected format/shape.
    """
    inchi = getattr(pyg_data, 'inchi', 'N/A') # Get InChI for better error context

    # Early exit if there are no nodes, as variable-length properties often relate to atoms/structure
    if pyg_data.num_nodes == 0:
        raise PropertyEnrichmentError(
            molecule_index=molecule_index,
            inchi=inchi,
            reason="No nodes found for variable-length graph properties."
        )

    for key in property_keys:
        try:
            # --- Direct access from raw_properties_dict ---
            value = raw_properties_dict.get(key)

            # General validity check for all variable-length properties first
            if not _is_value_valid_and_not_nan(value):
                raise PropertyEnrichmentError(
                    molecule_index=molecule_index,
                    inchi=inchi,
                    property_name=key, # Add property_name
                    reason=f"Missing, invalid, or NaN variable-length property '{key}'."
                )

            if key == 'vibmodes':
                num_atoms = pyg_data.num_nodes # Get number of atoms from the PyG data object

                if num_atoms == 0:
                    raise PropertyEnrichmentError(
                        molecule_index=molecule_index,
                        inchi=inchi,
                        property_name=key, # Add property_name
                        reason=f"Cannot process 'vibmodes' for index {molecule_index}: num_nodes is 0."
                    )

                # Ensure vibmodes data is a 3D NumPy array (num_modes, num_atoms, 3)
                reshaped_value = None
                if isinstance(value, np.ndarray) and value.ndim == 2 and value.shape[1] == 3:
                    # Case: (num_modes * num_atoms, 3)
                    if value.shape[0] % num_atoms != 0:
                        raise PropertyEnrichmentError(
                            molecule_index=molecule_index,
                            inchi=inchi,
                            property_name=key, # Add property_name
                            reason=f"'vibmodes' array has shape {value.shape}, but first dimension ({value.shape[0]}) is not a multiple of num_nodes ({num_atoms}). Cannot reshape to (num_modes, num_nodes, 3)."
                        )
                    num_modes = value.shape[0] // num_atoms
                    reshaped_value = value.reshape(num_modes, num_atoms, 3)
                    logger.debug(f"      Reshaped 'vibmodes' from {value.shape} to {reshaped_value.shape} for index {molecule_index}.")
                elif isinstance(value, np.ndarray) and value.ndim == 3 and value.shape[2] == 3 and value.shape[1] == num_atoms:
                    # Case: Already (num_modes, num_atoms, 3)
                    reshaped_value = value
                    logger.debug(f"      'vibmodes' for index {molecule_index} is already in expected 3D format: {value.shape}.")
                else:
                    raise PropertyEnrichmentError(
                        molecule_index=molecule_index,
                        inchi=inchi,
                        property_name=key, # Add property_name
                        reason=f"'vibmodes' array has unexpected NumPy array format or type: {value.shape if isinstance(value, np.ndarray) else type(value)}."
                    )

                # Store vibmodes as a Python list of tensors, one tensor per mode.
                # This prevents PyTorch Geometric's `collate` from attempting to `torch.cat`
                # across the varying `num_modes` dimension, avoiding the `RuntimeError`.
                vibmodes_list_of_tensors = [torch.tensor(mode_data, dtype=torch.float) for mode_data in reshaped_value]
                setattr(pyg_data, key, vibmodes_list_of_tensors)

                continue # Skip to the next property_key in the loop

            # This block handles all *other* variable-length properties (e.g., 'freqs')
            dtype = torch.complex64 if 'freq' in key else torch.float
            setattr(pyg_data, key, torch.tensor(value, dtype=dtype))

        # Catch PropertyEnrichmentError explicitly, then general Exception
        except PropertyEnrichmentError:
            raise # Re-raise our specific error
        except Exception as e:
            raise PropertyEnrichmentError(
                molecule_index=molecule_index,
                inchi=inchi,
                property_name=key, # Add property_name
                reason=f"Error fetching variable-length property '{key}'.",
                detail=str(e)
            ) from e


def calculate_atomization_energy(
    molecular_total_energy_hartree: float,
    atomic_numbers_tensor: torch.Tensor,
    molecule_index: int,
    inchi: str,
    logger: logging.Logger
) -> float:
    """
    Calculates the atomization energy of a molecule in electronvolts (eV).

    This is computed as the difference between the sum of the energies of its
    isolated atoms and the molecule's total energy. Both are converted to eV
    from Hartree using `HAR2EV`.

    Args:
        molecular_total_energy_hartree (float): The total energy of the molecule in Hartree.
        atomic_numbers_tensor (torch.Tensor): A 1D tensor containing the atomic numbers (Z)
                                               for all atoms in the molecule.
        molecule_index (int): The unique index of the molecule being processed.
        inchi (str): The InChI string representation of the molecule.
        logger (logging.Logger): The logger instance for recording messages.

    Returns:
        float: The calculated atomization energy in eV.

    Raises:
        PropertyEnrichmentError: If the atomic energy for any constituent atom
                                 is not found in `ATOMIC_ENERGIES_HARTREE`.
        ConfigurationError: If `HAR2EV` or `ATOMIC_ENERGIES_HARTREE` are
                            missing or improperly defined in the `config` module.
    """
    sum_atomic_energies_hartree = 0.0

    # Add checks for HAR2EV and ATOMIC_ENERGIES_HARTREE from config
    if HAR2EV is None:
        raise ConfigurationError(
            message="HAR2EV (Hartree to eV conversion factor) is not defined in config.",
            config_key="HAR2EV"
        )
    if not ATOMIC_ENERGIES_HARTREE:
        raise ConfigurationError(
            message="ATOMIC_ENERGIES_HARTREE is empty or not defined in config.",
            config_key="ATOMIC_ENERGIES_HARTREE"
        )


    for atomic_num in atomic_numbers_tensor.tolist():
        atomic_energy = ATOMIC_ENERGIES_HARTREE.get(atomic_num)
        if atomic_energy is None:
            raise PropertyEnrichmentError(
                molecule_index=molecule_index,
                inchi=inchi,
                reason=f"Missing atomic energy for atomic number {atomic_num}. Cannot calculate atomization energy."
            )
        sum_atomic_energies_hartree += atomic_energy

    # Convert both molecular and summed atomic energies to eV before subtraction
    molecular_total_energy_eV = molecular_total_energy_hartree * HAR2EV
    sum_atomic_energies_eV = sum_atomic_energies_hartree * HAR2EV

    atomization_energy_eV = molecular_total_energy_eV - sum_atomic_energies_eV
    return atomization_energy_eV


def add_derived_graph_targets(
    pyg_data: Data,
    raw_properties_dict: Dict[str, Union[float, int, np.number, np.ndarray, None]],
    molecule_index: int,
    logger: logging.Logger,
    data_config: Dict[str, Any]
) -> None:
    """
    Calculates and appends derived scalar graph-level targets to `pyg_data.y`.

    Currently supports the calculation of atomization energy if configured.
    Ensures that required input data for derived properties is valid and available.

    Args:
        pyg_data (torch_geometric.data.Data): The PyG Data object for the current molecule.
        raw_properties_dict (dict): A dictionary containing all raw data extracted for the molecule.
        molecule_index (int): The index of the molecule being processed.
        logger (logging.Logger): The logger instance.
        data_config (dict): The configuration dictionary for data processing,
                            including parameters for derived properties (e.g.,
                            'calculate_atomization_energy_from', 'atomization_energy_key_name').

    Raises:
        PropertyEnrichmentError: If the calculation fails due to missing or invalid
                                 input data, or unexpected data types/formats.
        ConfigurationError: If essential configuration for derived properties is missing or invalid.
    """
    inchi = getattr(pyg_data, 'inchi', 'N/A') # Get InChI for better error context

    # Atomization energy calculation
    atomization_energy_base_key = data_config.get('calculate_atomization_energy_from')
    atomization_energy_output_key = data_config.get('atomization_energy_key_name')

    if atomization_energy_base_key and atomization_energy_output_key:
        # --- Get base energy directly from raw_properties_dict ---
        molecular_total_energy_hartree = raw_properties_dict.get(atomization_energy_base_key)

        if not _is_value_valid_and_not_nan(molecular_total_energy_hartree):
            raise PropertyEnrichmentError(
                molecule_index=molecule_index,
                inchi=inchi,
                property_name=atomization_energy_base_key, # Add property_name
                reason=f"Cannot calculate atomization energy: Base energy '{atomization_energy_base_key}' is missing or invalid."
            )
        # Ensure it's a float scalar
        if isinstance(molecular_total_energy_hartree, np.ndarray) and molecular_total_energy_hartree.size == 1:
            molecular_total_energy_hartree = float(molecular_total_energy_hartree.item())
        elif not isinstance(molecular_total_energy_hartree, (int, float, np.number)):
            raise PropertyEnrichmentError(
                molecule_index=molecule_index,
                inchi=inchi,
                property_name=atomization_energy_base_key, # Add property_name
                reason=f"Cannot calculate atomization energy: Base energy '{atomization_energy_base_key}' has unexpected type {type(molecular_total_energy_hartree)}."
            )

        # Ensure atomic numbers (z) are available and non-empty on pyg_data
        if not hasattr(pyg_data, 'z') or pyg_data.z is None or pyg_data.z.numel() == 0:
            raise PropertyEnrichmentError(
                molecule_index=molecule_index,
                inchi=inchi,
                reason=f"Missing, None, or empty 'z' (atomic numbers). Cannot calculate atomization energy."
            )

        try:
            atomization_e = calculate_atomization_energy(
                molecular_total_energy_hartree,
                pyg_data.z,
                molecule_index, # Pass molecule_index
                inchi,          # Pass inchi
                logger
            )

            if np.isnan(atomization_e): # Check for NaN explicitly
                # This block should ideally not be reached if calculate_atomization_energy raises an error
                # but kept as a defensive check.
                raise PropertyEnrichmentError(
                    molecule_index=molecule_index,
                    inchi=inchi,
                    reason=f"Calculated atomization energy is NaN (likely due to missing atomic energy data for one or more atoms)."
                )

            # --- Ensure pyg_data.y exists before concatenating ---
            # If scalar_graph_targets_to_include was empty, pyg_data.y might not exist.
            if not hasattr(pyg_data, 'y') or pyg_data.y is None or pyg_data.y.numel() == 0:
                pyg_data.y = torch.tensor([atomization_e], dtype=torch.float)
            else:
                pyg_data.y = torch.cat([pyg_data.y, torch.tensor([atomization_e], dtype=torch.float)])
            logger.debug(f"      Calculated atomization energy ({atomization_energy_output_key}): {atomization_e:.4f} eV for index {molecule_index}")

        # Catch specific exceptions from calculate_atomization_energy
        except (PropertyEnrichmentError, ConfigurationError):
            raise # Re-raise our specific errors
        except Exception as e:
            # Catch any other unexpected errors during calculation
            raise PropertyEnrichmentError(
                molecule_index=molecule_index,
                inchi=inchi,
                reason=f"Error calculating atomization energy.",
                detail=str(e)
            ) from e


def enrich_pyg_data_with_properties(
    pyg_data: Data,
    mol_idx: int,
    raw_properties_dict: Dict[str, Any],
    inchi_identifier: str, # MODIFIED LINE
    logger: logging.Logger,
    data_config: Optional[Dict[str, Any]] = None,
) -> Data:
    """
    Orchestrates the enrichment of a PyG Data object with various molecular properties.

    This function calls helper functions to add scalar graph targets (`pyg_data.y`),
    node features (`pyg_data.x`), graph-level vector properties (e.g., `pyg_data.dipole`),
    and variable-length graph properties (e.g., `pyg_data.freqs`, `pyg_data.vibmodes`).
    It also handles the calculation of derived properties like atomization energy.

    Exceptions (`PropertyEnrichmentError`, `ConfigurationError`) raised by helper functions
    are caught and re-raised here to allow the upstream data processing pipeline to
    handle molecules that fail enrichment gracefully (e.g., by skipping them).

    Args:
        pyg_data (torch_geometric.data.Data): The PyG Data object to be enriched.
        mol_idx (int): The unique index of the current molecule.
        raw_properties_dict (dict): A dictionary containing all pre-extracted raw data
                                    for the current molecule, accessible by property key.
        inchi_identifier (str): The InChI string of the current molecule, used for error context. # MODIFIED LINE
        logger (logging.Logger): The logger instance for recording messages.
        data_config (dict, optional): A dictionary specifying which properties to include
                                      or derive. Defaults to an empty dictionary if None.

    Returns:
        torch_geometric.data.Data: The `pyg_data` object, now enriched with the specified properties.

    Raises:
        PropertyEnrichmentError: If any property is missing, invalid, or malformed during
                                 the enrichment process, or if an unexpected error occurs.
        ConfigurationError: If essential configuration parameters required for enrichment
                            (e.g., for atomization energy calculation) are missing or invalid.
    """
    # Retrieve configurations from data_config inside the function
    data_config = data_config if data_config is not None else {}
    scalar_graph_targets_to_include = data_config.get('scalar_graph_targets_to_include', [])
    node_features_to_add = data_config.get('node_features_to_add', [])
    vector_graph_properties_to_include = data_config.get('vector_graph_properties_to_include', [])
    variable_len_graph_properties_to_include = data_config.get('variable_len_graph_properties_to_include', [])

    # The is_complete_ref list is completely removed from this function's scope.
    # We now rely solely on exceptions.

    try:
        # 1. Add scalar graph targets
        # Pass raw_properties_dict
        add_scalar_graph_targets(pyg_data, raw_properties_dict, mol_idx, logger, scalar_graph_targets_to_include)

        # 2. Add node features
        # This line ensures num_nodes is available before add_node_features
        # If pyg_data.z is not set by an upstream step (e.g., mol_to_pyg_data),
        # this might raise an AttributeError. The general Exception catch will handle it.
        pyg_data.num_nodes = pyg_data.z.size(0) if hasattr(pyg_data, 'z') and pyg_data.z is not None else \
                             pyg_data.x.size(0) if hasattr(pyg_data, 'x') and pyg_data.x is not None else 0

        # Early exit check for 0 nodes, which should already be covered by add_node_features
        # but as a sanity check here:
        if pyg_data.num_nodes == 0:
            raise PropertyEnrichmentError(
                molecule_index=mol_idx,
                inchi=inchi_identifier, # MODIFIED LINE
                reason="PyG data object has 0 nodes after initial processing."
            )

        # Pass raw_properties_dict
        add_node_features(pyg_data, raw_properties_dict, mol_idx, logger, node_features_to_add)

        # 3. Call add_derived_graph_targets after base properties are added
        # This function also handles its own internal checks and raises errors.
        # Pass raw_properties_dict
        add_derived_graph_targets(pyg_data, raw_properties_dict, mol_idx, logger, data_config)

        # 4. Add vector graph properties
        # Pass raw_properties_dict
        add_vector_graph_properties(pyg_data, mol_idx, raw_properties_dict, vector_graph_properties_to_include, inchi_identifier, logger) 

        # 5. Add variable-length graph properties
        # Pass raw_properties_dict
        add_variable_len_graph_properties(pyg_data, raw_properties_dict, mol_idx, logger, variable_len_graph_properties_to_include)

    # Catch specific PropertyEnrichmentError and ConfigurationError
    except (PropertyEnrichmentError, ConfigurationError):
        # If any of the called functions raises our specific errors,
        # we simply re-raise them. The caller (e.g., VQM24Dataset.process)
        # will then catch it and handle the incomplete molecule.
        logger.warning(f"Molecule {mol_idx} {inchi_identifier} processing failed at enrichment stage. Propagating error.") 
        raise

    except Exception as e:
        # Catch any other unexpected errors during the enrichment process itself
        # that weren't caught by the specific property functions.
        logger.error(f"Unexpected error during enrichment for molecule {mol_idx} (InChI: {inchi_identifier}): {e}", exc_info=True) 
        raise PropertyEnrichmentError(
            molecule_index=mol_idx,
            inchi=inchi_identifier, 
            reason="Unexpected error during property enrichment.",
            detail=str(e)
        ) from e

    return pyg_data
