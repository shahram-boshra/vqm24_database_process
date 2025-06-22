# data_utils.py

import logging
import numpy as np
import torch
from typing import Any, Union, Dict

"""
Utility functions for data processing and validation.

This module provides helper functions for validating data values and safely
accessing elements within preloaded NumPy arrays, with robust error handling.
"""
from exceptions import MoleculeProcessingError, MoleculeFilterRejectedError, DataProcessingError, ConfigurationError


logger = logging.getLogger(__name__)


def _is_value_valid_and_not_nan(value: Any) -> bool:
    """
    Checks if a given value is valid and not NaN.

    Args:
        value: The value to check. Can be of various types, including None,
               str, bytes, numpy strings/bytes, or numerical types that can
               be converted to a numpy array.

    Returns:
        bool: True if the value is valid and does not contain NaN, False otherwise.

    Raises:
        DataProcessingError: If the value cannot be converted to a NumPy array
                             or contains invalid data for array conversion.
    """
    if value is None: return False
    if isinstance(value, (str, bytes, np.str_, np.bytes_)): return bool(str(value).strip())
    try:
        arr_value: np.ndarray = np.asarray(value)
    except TypeError as e: # Catch a more specific error for array conversion
        # Raise a DataProcessingError for an invalid value type
        raise DataProcessingError(
            message="Value cannot be converted to a NumPy array.",
            item_identifier=f"Value: {value}",
            details=f"Original error: {e}"
        ) from e # Use 'from e' to chain exceptions
    except ValueError as e: # Another common error for invalid conversions
        raise DataProcessingError(
            message="Value contains invalid data for array conversion.",
            item_identifier=f"Value: {value}",
            details=f"Original error: {e}"
        ) from e
    except Exception as e: # Catch any other unexpected conversion issues
        raise DataProcessingError(
            message="An unexpected error occurred during array conversion.",
            item_identifier=f"Value: {value}",
            details=f"Original error: {e}"
        ) from e

    if arr_value.size == 0 and arr_value.ndim > 0: return False # Empty non-scalar array
    if np.issubdtype(arr_value.dtype, np.floating): return not np.any(np.isnan(arr_value))
    elif np.issubdtype(arr_value.dtype, np.complexfloating): return not (np.any(np.isnan(arr_value.real)) or np.any(np.isnan(arr_value.imag)))
    elif arr_value.dtype == object:
        for x in arr_value.flat:
            if not _is_value_valid_and_not_nan(x): return False
    return True


def get_array_element(preloaded_data_dict: Dict[str, np.ndarray], array_key: str, index: int) -> Any:
    """
    Retrieves a specific element from a NumPy array stored in a dictionary.

    Args:
        preloaded_data_dict (Dict[str, np.ndarray]): A dictionary where keys are strings
                                                     and values are NumPy arrays.
        array_key (str): The key corresponding to the desired NumPy array in the dictionary.
        index (int): The integer index of the element to retrieve from the array.

    Returns:
        Any: The element at the specified index from the array. The type can vary
             depending on the array's dtype (e.g., int, float, str, etc.).

    Raises:
        DataProcessingError: If the array_key is not found, the index is out of bounds,
                             or any other error occurs during array access.
    """
    try:
        array: np.ndarray = preloaded_data_dict[array_key]
        if index < 0 or index >= array.shape[0]:
            # Wrap IndexError in DataProcessingError
            raise DataProcessingError(
                message="Attempted to access array element with an out-of-bounds index.",
                item_identifier=f"Array: '{array_key}'",
                details=f"Index {index} out of bounds for array of shape {array.shape}"
            )

        if array.ndim == 0: # Handle scalar numpy arrays (e.g., from .item())
            if index == 0:
                return array.item()
            else:
                # If scalar array and index is not 0, it's an invalid access
                raise DataProcessingError(
                    message="Attempted to access non-existent element in a scalar array.",
                    item_identifier=f"Array: '{array_key}'",
                    details=f"Scalar array accessed with index {index}. Only index 0 is valid for scalar arrays."
                )
        else:
            return array[index]
    except KeyError as e:
        # Wrap KeyError in DataProcessingError
        raise DataProcessingError(
            message="Required array key not found in preloaded data.",
            item_identifier=f"Missing key: '{array_key}'",
            details=f"Original error: {e}"
        ) from e
    except Exception as e:
        # Catch any other unexpected errors during array access
        raise DataProcessingError(
            message="An unexpected error occurred while getting an array element.",
            item_identifier=f"Array: '{array_key}', Index: {index}",
            details=f"Original error: {e}"
        ) from e
    
