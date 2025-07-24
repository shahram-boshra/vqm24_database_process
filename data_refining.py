# data_refining.py

"""
This module provides functions for refining molecular vibrational data,
specifically frequencies and their corresponding vibrational modes.

It includes utilities for:
- Validating and cleaning numerical values, including handling NaN and various array types.
- Deeply converting nested data structures to float.
- Flattening nested lists of numeric types.
- Normalizing individual vibrational modes to a consistent (N_atoms, 3) numpy array format.
- Refining a set of molecular frequencies and vibmodes by removing invalid or near-zero entries,
  and identifying and retaining only unique (frequency, vibmode) pairs.

The module aims to ensure data quality and consistency for downstream processing
of vibrational analysis results.
"""

import numpy as np
import logging
import torch
from typing import Any, List, Union

from exceptions import VibrationRefinementError, DataProcessingError

logger = logging.getLogger(__name__)


def _is_value_valid_and_not_nan(value, allow_empty_array: bool = False) -> bool:
    """
    Checks if a given value is valid (not None, not NaN) and if it's a numeric
    type or an array containing only numeric, non-NaN values. Handles object dtypes
    in numpy arrays by recursively checking elements.
    """
    if value is None:
        return False

    if isinstance(value, (int, float, np.number)):
        return not np.isnan(value)

    if isinstance(value, (np.ndarray, list)):
        value_np = np.array(value, dtype=object) # Use object to avoid immediate conversion issues

        if value_np.size == 0:
            return allow_empty_array

        # Check if the array itself is numeric or can be safely cast
        if np.issubdtype(value_np.dtype, np.number):
            return not np.any(np.isnan(value_np))
        elif value_np.dtype == object:
            # If dtype is object, iterate and check each element
            for element in value_np.flat:
                # Ensure elements are valid and numeric
                if not isinstance(element, (int, float, np.number, np.ndarray)) or \
                   (isinstance(element, np.ndarray) and not np.issubdtype(element.dtype, np.number)) or \
                   not _is_value_valid_and_not_nan(element, allow_empty_array=allow_empty_array):
                    return False
            return True
        else:
            # Non-numeric dtype that is not object (e.g., boolean, string arrays)
            return False
    return False


def _deep_convert_to_float(item: Any) -> Union[float, List[float], None]:
    """
    Recursively converts elements of a nested list/array structure to float.
    Returns None if any element cannot be converted.
    """
    if isinstance(item, (int, float, np.number)):
        return float(item)
    elif isinstance(item, (list, np.ndarray)):
        converted_list = []
        for sub_item in item:
            converted_sub_item = _deep_convert_to_float(sub_item)
            if converted_sub_item is None:
                return None  # Propagate failure
            if isinstance(converted_sub_item, list):
                converted_list.extend(converted_sub_item)
            else:
                converted_list.append(converted_sub_item)
        return converted_list
    else:
        logger.warning(f"Unsupported type encountered during deep conversion to float: {type(item).__name__} with value '{item}'. Returning None.")
        return None


def _flatten_list_if_nested(input_list: List[Any]) -> List[float]:
    """
    Flattens a list of lists into a single list of floats.
    Assumes the deepest elements are numeric.
    """
    flattened = []
    for item in input_list:
        if isinstance(item, (list, np.ndarray)):
            # Recursively flatten if it's a list or numpy array
            flattened.extend(_flatten_list_if_nested(item))
        elif isinstance(item, (int, float, np.number)):
            flattened.append(float(item))
        else:
            logger.warning(f"Non-numeric item encountered during flattening: {type(item).__name__} = {item}")
    return flattened


def _normalize_vibmode(vibmode_entry: Any, molecule_index: int = -1) -> np.ndarray:
    """
    Normalizes a single vibmode entry to (N_atoms, 3) and ensures it's a numeric array.
    Handles various input formats including nested lists.

    Raises:
        VibrationRefinementError: If the vibmode cannot be normalized or is invalid.
    """
    if vibmode_entry is None:
        raise VibrationRefinementError(
            message="Received None for vibmode_entry.",
            molecule_index=molecule_index,
            reason="Vibmode entry is None."
        )

    # First, attempt a deep conversion to ensure all elements are floats
    logger.debug(f"DEBUG: _normalize_vibmode - Before deep_convert_to_float. Input type: {type(vibmode_entry)}, value (truncated): {str(vibmode_entry)[:100]}") # Truncate for logging
    processed_vibmode_entry = _deep_convert_to_float(vibmode_entry)
    logger.debug(f"DEBUG: _normalize_vibmode - After deep_convert_to_float. Processed type: {type(processed_vibmode_entry)}, value (truncated): {str(processed_vibmode_entry)[:100]}") # Truncate for logging

    if processed_vibmode_entry is None:
        raise VibrationRefinementError(
            message="Deep conversion to float failed for vibmode entry.",
            molecule_index=molecule_index,
            reason="Failed to convert vibmode elements to float."
        )

    try:
        # Attempt to convert to numpy array of float64
        # This will handle lists of floats or single floats directly
        normalized_vibmode = np.array(processed_vibmode_entry, dtype=np.float64)
    except ValueError as ve:
        raise VibrationRefinementError(
            message="Could not convert processed vibmode entry to a numeric numpy array.",
            molecule_index=molecule_index,
            reason="Invalid data type for vibmode elements.",
            detail=str(ve)
        ) from ve
    except Exception as e:
        raise VibrationRefinementError(
            message="An unexpected error occurred during vibmode conversion after deep processing.",
            molecule_index=molecule_index,
            reason="Unexpected error during numpy array conversion.",
            detail=str(e)
        ) from e

    # Ensure it's numeric and not empty
    if not np.issubdtype(normalized_vibmode.dtype, np.number) or normalized_vibmode.size == 0:
        raise VibrationRefinementError(
            message="Normalized vibmode is not numeric or is empty after conversion.",
            molecule_index=molecule_index,
            reason=f"Vibmode Dtype: {normalized_vibmode.dtype}, Size: {normalized_vibmode.size}."
        )

    # Reshape if it's a 1D array, inferring N_atoms
    # Assuming the total number of elements must be divisible by 3 (N_atoms * 3 coordinates)
    if normalized_vibmode.ndim == 1:
        if normalized_vibmode.size % 3 == 0:
            n_atoms = normalized_vibmode.size // 3
            normalized_vibmode = normalized_vibmode.reshape(n_atoms, 3)
            logger.debug(f"Reshaped 1D vibmode to ({n_atoms}, 3) successfully.")
        else:
            raise VibrationRefinementError(
                message=f"1D vibmode size ({normalized_vibmode.size}) is not divisible by 3.",
                molecule_index=molecule_index,
                reason="Cannot reshape 1D vibmode to (N_atoms, 3)."
            )
    elif normalized_vibmode.ndim == 2 and normalized_vibmode.shape[1] == 3:
        # Already in (N_atoms, 3) format
        pass
    else:
        raise VibrationRefinementError(
            message=f"Vibmode has unexpected shape {normalized_vibmode.shape}.",
            molecule_index=molecule_index,
            reason="Expected 1D or (N_atoms, 3) vibmode array."
        )

    return normalized_vibmode

def log_vibration_refinement_status(
    raw_freqs_data: Any,
    raw_vibmodes_data: Any,
    molecule_index: int,
    logger: logging.Logger
) -> None:
    """
    Logs the status of vibrational data refinement based on the availability of
    frequencies and vibrational modes in the raw data.
    """
    freqs_available = raw_freqs_data is not None
    vibmodes_available = raw_vibmodes_data is not None

    if not freqs_available and not vibmodes_available:
        logger.info(f"Molecule {molecule_index}: Skipping vibrational data refinement: 'freqs' and 'vibmodes' are both not chosen to be processed.")
    elif not freqs_available:
        logger.info(f"Molecule {molecule_index}: Skipping vibrational frequencies data refinement: 'freqs' is not not chosen to be processed.")
    elif not vibmodes_available:
        logger.info(f"Molecule {molecule_index}: Skipping vibrational modes data refinement: 'vibmodes' is not chosen to be processed.")
    else:
        pass

def refine_molecular_vibrations(freqs: np.ndarray, vibmodes: np.ndarray, comparison_tolerance: float = 1e-4, molecule_index: int = -1):
    """
    Refines vibrational frequencies and modes for a single molecule by:
    1. Removing empty or near-zero frequencies and their corresponding vibmodes.
    2. Identifying and keeping only unique (frequency, vibmode) pairs.
    3. Ensuring all vibmodes are reshaped to (N_atoms, 3) if they are 1D arrays,
       and are always of a numeric dtype.

    Args:
        freqs (np.ndarray): An array of complex frequencies for a single molecule.
        vibmodes (np.ndarray): An array of vibrational modes (displacements)
                                corresponding to the frequencies. Can contain
                                nested lists or arrays of varying depths.
        comparison_tolerance (float): The absolute tolerance for numerical
                                      comparisons (e.g., np.isclose) when checking
                                      for zero values or duplicates.
        molecule_index (int): The index of the molecule being processed. Used for debugging.

    Returns:
        tuple: A tuple containing:
            - cleaned_freqs (list): List of refined (non-zero, unique) frequencies.
            - cleaned_vibmodes (list): List of refined (non-zero, unique) vibmodes.
            - is_accepted (bool): True if the number of cleaned frequencies equals
                                  the number of cleaned vibmodes, indicating a valid
                                  1:1 correspondence.

    Raises:
        VibrationRefinementError: If no valid frequency-vibmode pairs are found,
                                  or if there's a mismatch in the counts after refinement.
    """
    # Step 1: Remove empty or near-zero frequencies and their corresponding vibmodes
    # Also filter out vibmodes that cannot be normalized
    cleaned_freqs = []
    cleaned_vibmodes = []

    for i, freq_entry in enumerate(freqs):
        # Check if frequency is valid and not near zero
        is_freq_valid = _is_value_valid_and_not_nan(freq_entry) and not np.isclose(freq_entry, 0.0, atol=comparison_tolerance)

        if is_freq_valid:
            vibmode_entry = vibmodes[i]
            try:
                normalized_vibmode = _normalize_vibmode(vibmode_entry, molecule_index)

                # Check if normalized_vibmode is valid and not empty
                if _is_value_valid_and_not_nan(normalized_vibmode, allow_empty_array=False) and normalized_vibmode.size > 0:
                    cleaned_freqs.append(freq_entry)
                    cleaned_vibmodes.append(normalized_vibmode)
                else:
                    logger.warning(f"Molecule {molecule_index}: Normalized vibmode at index {i} is not valid or is empty. Skipping this pair.")
            except VibrationRefinementError as e:
                logger.warning(f"Molecule {molecule_index}: Skipping vibmode at index {i} due to normalization error: {e}")
        else:
            logger.warning(f"Molecule {molecule_index}: Frequency at index {i} is invalid or near zero ({freq_entry}). Skipping this pair.")


    if not cleaned_freqs:
        raise VibrationRefinementError(
            message="No valid frequency-vibmode pairs found after initial filtering.",
            molecule_index=molecule_index,
            reason="All frequency-vibmode pairs were invalid or zero."
        )

    # Step 2: Identify and keep only unique (frequency, vibmode) pairs
    # Use a set to store string representations of (frequency, vibmode) for uniqueness check
    unique_pairs = set()
    final_unique_freqs = []
    final_unique_vibmodes = []

    for freq, vibmode in zip(cleaned_freqs, cleaned_vibmodes):
        # Create a stable string representation for the vibmode for hashing
        vibmode_str = np.array2string(vibmode, precision=6, separator=',', suppress_small=True)
        pair_str = f"{freq},{vibmode_str}"

        if pair_str not in unique_pairs:
            unique_pairs.add(pair_str)
            final_unique_freqs.append(freq)
            final_unique_vibmodes.append(vibmode)

    # Step 3: Determine acceptance based on count parity and log
    is_accepted = (len(final_unique_freqs) == len(final_unique_vibmodes))

    if is_accepted:
        logger.info(f"Molecule {molecule_index}: Refinement complete: Accepted. Number of unique freqs = {len(final_unique_freqs)}, Number of unique vibmodes = {len(final_unique_vibmodes)}")
    else:
        reason_msg = f"Mismatch in counts: freqs = {len(final_unique_freqs)}, vibmodes = {len(final_unique_vibmodes)}."
        raise VibrationRefinementError(
            message="Refinement complete: Rejected. Mismatch in counts.",
            molecule_index=molecule_index,
            reason=reason_msg
        )

    return final_unique_freqs, final_unique_vibmodes, is_accepted
