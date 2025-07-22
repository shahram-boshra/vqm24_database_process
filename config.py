"""
This module handles the loading and access of application configuration settings
from a YAML file.

It ensures configuration integrity by validating keys and types, and provides
convenient access to frequently used constants.
"""
import yaml
import os
from typing import Dict, Any, Optional, Union, Tuple


from exceptions import ConfigurationError, AtomFilterError


# Global variable to store the loaded config
_CONFIG: Optional[Dict[str, Any]] = None
"""Stores the loaded application configuration as a dictionary,
cached after the first call to `load_config`."""

def load_config(config_path: str = 'config.yaml') -> Dict[str, Any]:
    """
    Loads the application configuration from the specified YAML file.

    Caches the configuration to ensure it's loaded only once per application run.

    Args:
        config_path (str): The path to the YAML configuration file.
                            Defaults to 'config.yaml'.

    Returns:
        dict: A dictionary containing the loaded configuration.

    Raises:
        ConfigurationError: If the file is not found, cannot be parsed,
                            or any other loading error occurs.
    """
    global _CONFIG
    if _CONFIG is None: # Load only once
        # (Optional: You can keep or remove this line, as it doesn't seem to be showing up for you)
        # print(f"DEBUG: Attempting to load config from: {os.path.abspath(config_path)}")

        if not os.path.exists(config_path):
            raise ConfigurationError(f"Configuration file not found at: {config_path}", config_key="config_path", actual_value=config_path)

        try:
            with open(config_path, 'r') as f:
                _CONFIG = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Error parsing configuration file: {e}", config_key="config_file", actual_value=config_path) from e
        except Exception as e:
            raise ConfigurationError(f"An unexpected error occurred while loading config: {e}", config_key="config_file", actual_value=config_path) from e
    return _CONFIG

def _get_config_value(config_dict: Dict[str, Any], key: str, expected_type: Optional[Union[type, tuple[type, ...]]] = None, parent_key: Optional[str] = None) -> Any:
    """
    Safely retrieves a value from the configuration dictionary with type validation.

    Args:
        config_dict (dict): The dictionary to retrieve the value from.
        key (str): The key to look up within the dictionary.
        expected_type (type or tuple[type, ...], optional): If provided,
            validates if the retrieved value is an instance of this type or any
            of the types in the tuple. Defaults to None (no type checking).
        parent_key (str, optional): The parent key, used for constructing
            more informative error messages (e.g., "section.key").

    Returns:
        Any: The value associated with the specified key.

    Raises:
        ConfigurationError: If the key is missing from the dictionary or
            the retrieved value does not match the `expected_type`.
    """
    full_key = f"{parent_key}.{key}" if parent_key else key
    if key not in config_dict:
        raise ConfigurationError(
            f"Missing required configuration key.",
            config_key=full_key
        )
    value = config_dict[key]
    if expected_type is not None and not isinstance(value, expected_type):
        raise ConfigurationError(
            f"Invalid type for configuration key '{full_key}'.",
            config_key=full_key,
            expected_type=expected_type,
            actual_value=value
        )
    return value

_TEMP_CONFIG: Dict[str, Any] = load_config()

# Make frequently used constants directly accessible from config.py
ATOMIC_ENERGIES_HARTREE: Dict[str, float] = _get_config_value(_TEMP_CONFIG, 'atomic_energies_hartree', dict)
HEAVY_ATOM_SYMBOLS_TO_Z: Dict[str, int] = _get_config_value(_TEMP_CONFIG, 'heavy_atom_symbols_to_z', dict)

# Access 'global_constants' dictionary first, then its 'har2ev' key
global_constants: Dict[str, Any] = _get_config_value(_TEMP_CONFIG, 'global_constants', dict)
HAR2EV: Union[int, float] = _get_config_value(global_constants, 'har2ev', (int, float), parent_key='global_constants')

# Dataset Configuration Constants
DATASET_CONFIG: Dict[str, Any] = _get_config_value(_TEMP_CONFIG, 'dataset_config', dict)
RAW_NPZ_FILENAME: str = _get_config_value(DATASET_CONFIG, 'raw_npz_filename', str)
RAW_NPZ_DOWNLOAD_URL: Optional[str] = _get_config_value(DATASET_CONFIG, 'raw_npz_download_url', (str, type(None))) # Allow str or None
DATASET_ROOT_DIR: str = _get_config_value(DATASET_CONFIG, 'dataset_root_dir', str)


