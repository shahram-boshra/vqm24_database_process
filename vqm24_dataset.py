# vqm24_dataset.py

"""
This module defines the `VQM24Dataset` class, an extension of `torch_geometric.data.InMemoryDataset`.

It handles the download, processing, and loading of the VQM24 dataset, including
support for chunked processing, pre-filtering of molecules based on configurable
criteria (e.g., atom counts, heavy atom presence), and application of PyG
pre-transformations. Robust error handling is implemented to manage issues
during data conversion and filtering.
"""

import logging
import os
import time
import sys
import shutil
from pathlib import Path
import numpy as np
import requests
from requests.exceptions import RequestException
import torch
import torch_geometric
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.transforms import Compose
from tqdm import tqdm
from typing import Optional, Dict, Any, List, Tuple, Union

import multiprocessing

from config import load_config
from data_utils import _is_value_valid_and_not_nan
from mol_conversion import MoleculeDataConverter
from molecule_filters import apply_pre_filters
from exceptions import (
    BaseProjectError,
    ConfigurationError,
    DataProcessingError,
    MoleculeProcessingError,
    MoleculeFilterRejectedError,
    MissingDependencyError,
    AtomFilterError,
    RDKitConversionError,
    PyGDataCreationError,
    PropertyEnrichmentError
)

logger = logging.getLogger(__name__)


def _delete_directory_in_background(directory_path_str: str, logger_name: str):
    """
    Deletes a directory in a separate process to avoid blocking the main thread.

    This is particularly useful for cleaning up large temporary directories
    (e.g., processed data chunks) after the main processing is complete.
    Accepts directory_path as a string because `Path` objects might not
    pickle reliably across processes in all Python versions/scenarios.

    Args:
        directory_path_str (str): The string representation of the path to the directory to be deleted.
        logger_name (str): The name of the logger to be used within the child process for logging messages.
    """
    # Re-initialize logger in the child process if detailed logging is desired
    # Note: This logger will operate independently of the parent's logger setup.
    process_logger = logging.getLogger(logger_name)

    directory_path = Path(directory_path_str) # Convert string back to Path object

    if directory_path.exists():
        process_logger.info(f"Background process: Deleting temporary directory: {directory_path}")
        try:
            shutil.rmtree(directory_path)
            process_logger.info(f"Background process: Deletion of {directory_path} completed.")
        except OSError as e:
            process_logger.error(f"Background process: Error deleting {directory_path}: {e}")
    else:
        process_logger.info(f"Background process: Directory {directory_path} not found, no deletion needed.")


class VQM24Dataset(InMemoryDataset):
    """
    VQM24Dataset is a PyTorch Geometric `InMemoryDataset` for the VQM24 quantum mechanics dataset.

    It handles the automatic download, conversion, and filtering of molecular data
    from an NPZ file into `torch_geometric.data.Data` objects. The dataset supports
    chunked processing to manage memory efficiently for large datasets, and allows
    for configurable pre-filtering (e.g., based on atom count, heavy atom presence)
    and custom PyTorch Geometric pre-transformations.

    The processed data is stored in a single `.pt` file for efficient loading
    in subsequent runs.

    Attributes:
        ZENODO_DOWNLOAD_URL (str): The URL to download the raw VQM24 NPZ file from Zenodo.

    Args:
        root (str): Root directory where the dataset should be saved. This directory
                    will contain `raw` and `processed` subdirectories.
        npz_file_path (str): The filename of the raw NPZ data file (e.g., "DFT_all.npz").
                              The dataset will attempt to download this file into `root/raw/`.
        data_config (Dict[str, Any]): Configuration dictionary specifying which properties
                                       to extract and how to handle them during molecule conversion.
        filter_config (Dict[str, Any]): Configuration dictionary for applying pre-filters
                                        to molecules (e.g., 'max_atoms', 'heavy_atom_filter').
        logger (logging.Logger): A logger instance for recording dataset-related messages.
        chunk_size (int, optional): The number of molecules to process before saving
                                    them to a temporary chunk file. Defaults to 5000.
                                    This helps manage memory during processing large datasets.
        transform (Optional[Any], optional): A function/transform that takes in a `torch_geometric.data.Data`
                                             object and returns a transformed version. Applied on the fly
                                             when accessing individual data points. Defaults to None.
        pre_transform (Optional[Any], optional): A function/transform that takes in a `torch_geometric.data.Data`
                                                 object and returns a transformed version. Applied once before
                                                 saving to disk. Defaults to None.
        pre_filter (Optional[Any], optional): A function that takes in a `torch_geometric.data.Data`
                                               object and returns a boolean value, indicating whether
                                               the data object should be included in the final dataset.
                                               Applied once before saving to disk. Defaults to None.
                                               Note: `apply_pre_filters` handles most filtering.
        pyg_pre_transforms_config (Optional[Dict[str, Any]], optional): Configuration for additional
                                                                         PyG `Compose` transforms to apply
                                                                         *before* saving the processed dataset.
                                                                         This is integrated into `self.pre_transform`.
        force_reload (bool, optional): If True, forces a re-download and re-processing of the dataset,
                                       even if processed files already exist. Defaults to False.
    """
    ZENODO_DOWNLOAD_URL: str = "https://zenodo.org/records/15442257/files/DFT_all.npz?download=1"

    def __init__(self,
                 root: str,
                 npz_file_path: str,
                 data_config: Dict[str, Any],
                 filter_config: Dict[str, Any],
                 structural_features_config: Dict[str, Any],
                 logger: logging.Logger,
                 chunk_size: int = 5000,
                 transform: Optional[Any] = None, # Type depends on PyG transform
                 pre_transform: Optional[Any] = None, # Type depends on PyG transform
                 pre_filter: Optional[Any] = None, # Type depends on PyG filter
                 pyg_pre_transforms_config: Optional[Dict[str, Any]] = None,
                 force_reload: bool = False):

        # Store only the filename, as the actual path will be downloaded
        self.raw_data_filename: str = Path(npz_file_path).name # Store just the filename
        self.data_config: Dict[str, Any] = data_config
        self.filter_config: Dict[str, Any] = filter_config
        self.structural_features_config: Dict[str, Any] = structural_features_config
        self.logger: logging.Logger = logger
        self.chunk_size: int = chunk_size
        self.force_reload: bool = force_reload
        self.processed_chunk_dir: Path = Path(root) / "processed_chunks"

        self.logger.info(f"Initializing VQM24Dataset with root: {root}, filters: {filter_config}, chunk_size: {chunk_size}")
        self.pre_transform_pipeline: Optional[Compose] = self._initialize_pre_transforms(pyg_pre_transforms_config)

        # NOTE: torch.serialization.add_safe_globals is deprecated/removed in PyTorch 2.x and later.
        # PyTorch Geometric Data objects are generally handled by PyTorch's default pickling
        # mechanisms without needing explicit registration for standard use cases.
        # If `torch.load` encounters issues with specific custom types within the
        # serialized data, a more advanced custom deserialization strategy
        # (e.g., using `dill` or custom `unpickler` in `torch.load`) might be needed.
        # Removed the following block as it caused an AttributeError:
        # safe_classes = [
        #     np.dtype,
        #     np.dtypes.StrDType, # These NumPy internal types might not exist or have changed in NumPy 2.x
        #     np._core.multiarray.scalar, # These NumPy internal types might not exist or have changed in NumPy 2.x
        #     torch_geometric.data.data.Data,
        #     torch_geometric.data.data.DataEdgeAttr,
        #     torch_geometric.data.data.DataTensorAttr,
        #     torch_geometric.data.storage.GlobalStorage,
        # ]
        # if Compose is not None:
        #     safe_classes.append(Compose)
        # torch.serialization.add_safe_globals(safe_classes) # This line is the direct cause of the error
        # self.logger.debug("Removed call to torch.serialization.add_safe_globals due to deprecation/removal.")


        super().__init__(root, transform, pre_transform=self.pre_transform_pipeline, force_reload=self.force_reload)

        processed_file_path: str = self.processed_paths[0]
        if Path(processed_file_path).exists():
            try:
                self.data: Optional[Data]
                self.slices: Optional[Dict[str, torch.Tensor]]
                self.data, self.slices = torch.load(processed_file_path)
                self.logger.info(f"Dataset data and slices loaded from {processed_file_path}.")
            except Exception as e: # Catching generic Exception here as torch.load can raise various errors
                self.logger.error(f"Error during manual load of {processed_file_path}: {e}")
                # Raise a specific DataProcessingError to signal this issue
                raise DataProcessingError(
                    message=f"Failed to load processed dataset from {processed_file_path}.",
                    details=f"Original error: {e.__class__.__name__}: {e}"
                ) from e # Chain the exception for better debugging
        else:
            self.logger.warning(f"Processed file {processed_file_path} does NOT exist after super().__init__. Processing might have failed to save it correctly, or all molecules were filtered out.")
            self.data, self.slices = None, None # It's okay for these to be None if no file was found, as process() will generate it.

        if isinstance(self.slices, dict) and self.slices:
            first_slice_key = next(iter(self.slices))
            inferred_len_from_slices = len(self.slices[first_slice_key])
            self.logger.debug(f"Length inferred from first slice key ('{first_slice_key}'): {inferred_len_from_slices}")
        else:
            self.logger.debug("self.slices is not a dictionary or is empty, cannot infer length directly from slices.")


        if len(self) == 0:
            self.logger.critical("Dataset is empty after processing/loading! Check errors during processing or if all molecules were filtered out.")
            self.logger.warning("Dataset is empty after processing/loading. Refer to ERROR/WARNING logs for details.")
        else:
            self.logger.info(f"Dataset successfully loaded/processed. Total molecules: {len(self)}")
            if hasattr(self.data, 'num_graphs'):
                self.logger.debug(f"self.data.num_graphs: {self.data.num_graphs}")
            elif isinstance(self.slices, dict) and self.slices:
                num_graphs_inferred = len(next(iter(self.slices.values())))
                self.logger.debug(f"Inferred number of graphs from slices: {num_graphs_inferred}")

    def _initialize_pre_transforms(self, pyg_pre_transforms_config: Optional[Dict[str, Any]]) -> Optional[Compose]:
        """
        Initializes and composes PyTorch Geometric pre-transformations based on configuration.

        This method dynamically loads and instantiates PyG transforms specified in the
        `pyg_pre_transforms_config`. It constructs a `torch_geometric.transforms.Compose`
        object that can be applied to `Data` objects during processing.

        Args:
            pyg_pre_transforms_config (Optional[Dict[str, Any]]): A dictionary containing
                the configuration for PyG pre-transforms. Expected keys:
                - 'enable' (bool): If True, enable transforms.
                - 'transforms' (List[Dict[str, Any]]): A list of dictionaries, each
                  describing a transform with 'name' and 'kwargs'.

        Returns:
            Optional[Compose]: A `Compose` object containing the initialized PyG transforms
                               if enabled and configured, otherwise None.

        Raises:
            ConfigurationError: If transform configuration is invalid (e.g., missing name,
                                invalid arguments, or no transforms found when enabled).
            MissingDependencyError: If a specified PyG transform class is not found.
            DataProcessingError: For unexpected errors during transform initialization.
        """
        if not pyg_pre_transforms_config or not pyg_pre_transforms_config.get('enable', False):
            self.logger.info("PyG pre-transformations are disabled or not configured.")
            return None

        transforms_list: List[Any] = [] # Can be various transform types
        for transform_info in pyg_pre_transforms_config.get('transforms', []):
            transform_name: str = transform_info.get('name')
            transform_kwargs: Dict[str, Any] = transform_info.get('kwargs', {})

            if not transform_name:
                raise ConfigurationError(
                    message="PyG transform entry missing 'name'.",
                    config_key="transform_name",
                    actual_value=transform_info
                )

            try:
                # Dynamically get the transform class from torch_geometric.transforms
                transform_class = getattr(torch_geometric.transforms, transform_name)
                transform_instance = transform_class(**transform_kwargs)
                transforms_list.append(transform_instance)
                self.logger.info(f"Successfully initialized PyG transform: {transform_name} with kwargs: {transform_kwargs}")
            except AttributeError:
                # Raise MissingDependencyError for transforms not found
                raise MissingDependencyError(
                    message=f"PyG transform '{transform_name}' not found in torch_geometric.transforms.",
                    dependency_name=f"torch_geometric.transforms.{transform_name}"
                )
            except TypeError as e:
                # Raise ConfigurationError for incorrect arguments
                raise ConfigurationError(
                    message=f"Error initializing PyG transform '{transform_name}' due to invalid arguments.",
                    config_key=f"transformations.transforms.{transform_name}.kwargs",
                    details=f"Original error: {e}"
                ) from e
            except Exception as e:
                # Catch any other unexpected errors during initialization
                raise DataProcessingError(
                    message=f"An unexpected error occurred during initialization of PyG transform '{transform_name}'.",
                    details=f"Error: {e.__class__.__name__}: {e}"
                ) from e

        if not transforms_list:
            # If transforms were enabled but none initialized successfully, it's a configuration issue.
            raise ConfigurationError(
                message="No valid PyG pre-transforms were initialized despite 'enable' being True.",
                config_key="transformations.transforms"
            )

        self.logger.info(f"Created PyG pre-transform pipeline with {len(transforms_list)} transforms.")
        return Compose(transforms_list)


    @property
    def raw_file_names(self) -> List[str]:
        """
        Returns a list of names of raw files in the `self.raw_dir` folder.

        This property is required by `torch_geometric.data.InMemoryDataset`.
        """
        # Use the stored filename
        return [self.raw_data_filename]

    @property
    def processed_file_names(self) -> List[str]:
        """
        Returns a list of names of processed files in the `self.processed_dir` folder.

        This property is required by `torch_geometric.data.InMemoryDataset`.
        The main processed data file is `data.pt`.
        """
        return ['data.pt']

    def download_file(
        url: str,
        filename: str,
        raw_dir: str,
        max_retries: int = 5,
        retry_delay: int = 5,
        logger: "logging.Logger" = None
        ) -> None:
        """
        Downloads a file from the specified URL and saves it to the given filename.

        Args:
            url (str): The URL of the file to download.
            filename (str): The name of the file to save.
            raw_dir (str): The directory to save the downloaded file.
            max_retries (int): The maximum number of retries in case of download failures.
            retry_delay (int): The initial delay (in seconds) between retries, with exponential backoff.
            logger (logging.Logger, optional): A logger instance to use for logging messages.

        Raises:
            RequestException: If the download fails after the maximum number of retries.
            IOError: If there is an error writing the downloaded data to the file.
            MissingDependencyError: If the `tqdm` library is not available.
        """
        destination_path: Path = Path(raw_dir) / filename

        # Check if file already exists and is not empty
        if destination_path.exists() and destination_path.stat().st_size > 0:
            if logger:
                logger.info(f"File already exists at {destination_path} and is non-empty. Skipping download.")
            return

        if logger:
            logger.info(f"Downloading file from {url} to {destination_path}...")

        retries = 0
        downloaded = 0
        block_size = 65536  # 64 KB

        while retries < max_retries:
            try:
                headers = {}
                if os.path.exists(destination_path):
                    downloaded = os.path.getsize(destination_path)
                    headers['Range'] = f'bytes={downloaded}-'

                with requests.get(url, stream=True, headers=headers) as response:
                    response.raise_for_status()
                    total_size_in_bytes = int(response.headers.get('content-length', 0))

                    with open(destination_path, 'ab') as file:
                        try:
                            with tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True, desc="Downloading", initial=downloaded) as pbar:
                                for data in response.iter_content(chunk_size=block_size):
                                    if data:
                                        file.write(data)
                                        downloaded += len(data)
                                        pbar.update(len(data))
                        except ModuleNotFoundError:
                            if logger:
                                logger.warning("tqdm library not found, progress bar will not be shown.")
                            with open(destination_path, 'ab') as file:
                                for data in response.iter_content(chunk_size=block_size):
                                    if data:
                                        file.write(data)
                                        downloaded += len(data)

                    if downloaded >= total_size_in_bytes:
                        if logger:
                            logger.info(f"Download complete: {destination_path}")
                        return
                    else:
                        if logger:
                            logger.warning(f"Download interrupted at {downloaded}/{total_size_in_bytes} bytes. Retrying...")

            except requests.exceptions.RequestException as e:
                if logger:
                    logger.error(f"Download error: {e}")
                retries += 1
                if retries < max_retries:
                    if logger:
                        logger.info(f"Retrying download in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    if logger:
                        logger.critical(f"Maximum number of retries reached. Unable to download '{filename}'.")
                    if os.path.exists(destination_path):
                        os.remove(destination_path)

    def process(self) -> None:
        """
        Processes the raw VQM24 dataset, converting molecules into PyG `Data` objects.

        This method performs the following steps:
        1. Loads required arrays (graphs, coordinates, and properties defined in `data_config`)
           from the raw NPZ file using memory mapping.
        2. Iterates through each molecule, converting it to a PyG `Data` object using
           `MoleculeDataConverter`.
        3. Applies configured PyG pre-transformations.
        4. Applies user-defined pre-filters (`apply_pre_filters`) to exclude molecules
           that do not meet specified criteria (e.g., atom limits, heavy atom composition).
           Molecules failing filters or conversion will be skipped, and relevant
           exceptions (e.g., `MoleculeFilterRejectedError`, `MoleculeProcessingError`)
           will be caught and logged.
        5. Saves processed molecules to temporary chunk files to manage memory.
        6. Concatenates all processed chunks into a single `torch_geometric.data.Data`
           object and saves it as `data.pt` in the `self.processed_dir`.
        7. Initiates a background process to clean up the temporary chunk directory.

        Raises:
            DataProcessingError: If the raw NPZ file is not found, or if essential
                                 keys are missing from the NPZ, or if there are issues
                                 during chunk loading/concatenation.
            RDKitConversionError: If RDKit-based molecule conversion fails for a molecule.
            PyGDataCreationError: If creating the PyG Data object or applying PyG
                                  pre-transforms fails for a molecule.
            PropertyEnrichmentError: If a specific property extraction or calculation fails.
            MoleculeFilterRejectedError: If a molecule is explicitly rejected by a filter.
            MoleculeProcessingError: For general errors during individual molecule processing.
            AtomFilterError: For configuration issues with the heavy atom filter.
        """
        self.logger.info("Starting VQM24Dataset processing (chunked mode)...")

        raw_npz_path_for_processing: str = self.raw_paths[0]

        self.logger.info(f"Pre-loading required arrays from {raw_npz_path_for_processing} into memory (mmap_mode='r')...")
        preloaded_data: Dict[str, np.ndarray] = {}
        # --- MODIFIED: Added 'inchi' and 'atoms' to all_required_keys ---
        all_required_keys: List[str] = ['graphs', 'coordinates', 'inchi', 'atoms'] # Explicitly list minimal required keys

        # Dynamically add all keys specified in data_config that might be needed
        # We need to collect ALL potential keys that any sub-function might try to access
        # from the raw data.
        for key_list_name in self.data_config:
            if isinstance(self.data_config[key_list_name], list):
                all_required_keys.extend(self.data_config[key_list_name])
            elif key_list_name in ['calculate_atomization_energy_from']:
                if self.data_config[key_list_name]: # Ensure it's not None or empty string
                    all_required_keys.append(self.data_config[key_list_name])

        all_required_keys = list(set(all_required_keys)) # Remove duplicates

        try:
            with np.load(raw_npz_path_for_processing, allow_pickle=True, mmap_mode='r') as data:
                # Use 'graphs' to determine total count, as confirmed by test_npzkeys.py
                total_molecules: int = data['graphs'].shape[0]
                test_molecule_limit: Optional[int] = self.data_config.get('test_molecule_limit')
                if test_molecule_limit is not None and test_molecule_limit > 0:
                    total_molecules = min(total_molecules, test_molecule_limit)
                    self.logger.warning(f"*** TEMPORARY: Limiting processing to {total_molecules} molecules for testing. Remove 'test_molecule_limit' from config for full dataset processing. ***")

                for key in all_required_keys:
                    if key in data.files:
                        preloaded_data[key] = data[key][:total_molecules] # Only load up to total_molecules
                    else:
                        self.logger.warning(f"    Key '{key}' not found in NPZ file. It will be skipped if requested by properties.")
            self.logger.info(f"Found {total_molecules} molecules and pre-loaded {len(preloaded_data)} arrays.")
        except FileNotFoundError:
            raise DataProcessingError(
                message=f"NPZ raw data file not found at '{raw_npz_path_for_processing}'.",
                details="Ensure the download step completed successfully."
            )
        except KeyError as e:
            raise DataProcessingError(
                message=f"Missing essential key in NPZ file: '{e}'. Cannot proceed with dataset processing.",
                details="The NPZ file might be corrupted or not in the expected format (e.g., 'graphs' array is missing)."
            ) from e
        except Exception as e:
            raise DataProcessingError(
                message=f"Failed to pre-load NPZ data from '{raw_npz_path_for_processing}'.",
                details=f"Original error: {e.__class__.__name__}: {e}"
            ) from e

        converter: MoleculeDataConverter = MoleculeDataConverter(
            self.logger,
            structural_features_config=self.structural_features_config
        ) # No longer pass preloaded_data here

        current_chunk_data_list: List[Data] = []
        processed_total_count: int = 0
        skipped_total_count: int = 0 # Consolidated count for all skips (conversion errors, filters, transforms)
        chunk_file_paths: List[Path] = []

        self.processed_chunk_dir.mkdir(parents=True, exist_ok=True)
        self.logger.debug(f"Processed data chunks will be saved to: {self.processed_chunk_dir}")

        for i in tqdm(range(total_molecules), desc="Processing Molecules"):
            pyg_data: Optional[Data] = None
            original_smiles: str = "N/A" # Default if fetching smiles fails

            # --- NEW: Create raw_properties_dict for the current molecule ---
            raw_properties_dict_for_current_mol: Dict[str, Any] = {} # Moved initialization inside loop
            try:
                # Populate raw_properties_dict with data for the current molecule index
                for key in all_required_keys:
                    if key in preloaded_data:
                        raw_properties_dict_for_current_mol[key] = preloaded_data[key][i]

                # Attempt to get InChI for robust error logging, as it's the primary identifier
                # and used for RDKit conversion.
                # Renamed from original_smiles to original_inchi for accuracy in logs.
                original_inchi_for_log = raw_properties_dict_for_current_mol.get('inchi', 'N/A')
                if original_inchi_for_log is None or original_inchi_for_log == 'N/A':
                    # Fallback if 'inchi' is somehow missing, use 'graphs' as a last resort
                    original_inchi_for_log = str(raw_properties_dict_for_current_mol.get('graphs', 'N/A'))
                    self.logger.debug(f"Molecule {i}: 'inchi' not found for logging, falling back to 'graphs' for InChI representation.")
                else:
                    original_inchi_for_log = str(original_inchi_for_log) # Ensure it's a string

                # --- Pass raw_properties_dict_for_current_mol to converter.convert ---
                pyg_data = converter.convert(i, raw_properties_dict_for_current_mol, self.data_config)

                if pyg_data is None:
                    # converter.convert already logs warnings/errors.
                    # This path means a non-MoleculeProcessingError issue occurred in converter.
                    self.logger.warning(f"Molecule {i} (InChI: {original_inchi_for_log}) conversion returned None without explicit error. Skipping.")
                    skipped_total_count += 1
                    continue # Skip to next molecule

                # --- Apply PyG Pre-transforms (if any) ---
                if self.pre_transform is not None:
                    try:
                        pyg_data = self.pre_transform(pyg_data)
                        self.logger.debug(f"Applied PyG pre_transform to molecule {i} (InChI: {original_inchi_for_log}).")
                    except Exception as e:
                        # Catch errors during PyG pre_transform application
                        raise PyGDataCreationError(
                            molecule_index=i,
                            smiles=original_smiles,
                            reason=f"Error applying PyG pre_transform: {e.__class__.__name__}",
                            detail=str(e)
                        ) from e

                # --- Apply user-defined pre-filters ---
                # apply_pre_filters now raises MoleculeFilterRejectedError for filter rejections
                apply_pre_filters(pyg_data, self.filter_config, self.logger)

                # If we reach here, the molecule passed all filters and conversions
                current_chunk_data_list.append(pyg_data)
                processed_total_count += 1

            # Exception block for MoleculeFilterRejectedError
            except MoleculeFilterRejectedError as e:
                # This block catches expected molecule rejections due to user-defined filters
                self.logger.info(f"Skipping molecule {e.molecule_index} (SMILES: {e.smiles}): {e.reason}")
                skipped_total_count += 1
            except RDKitConversionError as e: # Specific catch for RDKit conversion issues
                self.logger.warning(f"Skipping molecule {e.molecule_index} (SMILES: {e.smiles}): RDKit Conversion Failed: {e.reason} {e.detail}")
                skipped_total_count += 1
            except PyGDataCreationError as e: # Specific catch for PyG data object creation issues
                self.logger.warning(f"Skipping molecule {e.molecule_index} (SMILES: {e.smiles}): PyG Data Creation Failed: {e.reason} {e.detail}")
                skipped_total_count += 1
            except PropertyEnrichmentError as e: # Specific catch for issues during property enrichment
                self.logger.warning(f"Skipping molecule {e.molecule_index} (SMILES: {e.smiles}): Property Enrichment Failed for '{e.property_name}': {e.reason} {e.detail}")
                skipped_total_count += 1
            except MoleculeProcessingError as e:
                # This block catches all other expected processing failures (e.g., conversion issues, transform errors)
                self.logger.warning(f"Skipping molecule {e.molecule_index} (SMILES: {e.smiles}): {e.reason} {e.detail}")
                skipped_total_count += 1
            except Exception as e:
                # This block catches any unexpected critical errors not covered by specific exceptions
                self.logger.critical(f"CRITICAL UNHANDLED ERROR processing molecule {i} (InChI: {original_inchi_for_log}): {e.__class__.__name__} - {e}", exc_info=True)
                skipped_total_count += 1

            # Save chunks
            if (len(current_chunk_data_list) >= self.chunk_size) or \
               (i == total_molecules - 1 and len(current_chunk_data_list) > 0):

                chunk_idx: int = len(chunk_file_paths)
                chunk_filename: str = f'chunk_{chunk_idx:05d}.pt'
                chunk_path: Path = self.processed_chunk_dir / chunk_filename
                self.logger.info(f"Saving chunk {chunk_idx} with {len(current_chunk_data_list)} molecules to {chunk_path}...")
                torch.save(current_chunk_data_list, chunk_path)
                chunk_file_paths.append(chunk_path)
                current_chunk_data_list = []
                self.logger.debug(f"Chunk {chunk_idx} saved.")

        self.logger.info(f"Finished processing loop.")
        self.logger.info(f"Total molecules attempted: {total_molecules}")
        self.logger.info(f"Successfully processed and included: {processed_total_count} molecules.")
        self.logger.info(f"Skipped due to errors or filters: {skipped_total_count} molecules.")

        # Consolidate all chunks
        all_processed_data: List[Data] = []
        for chunk_path in tqdm(chunk_file_paths, desc="Loading and concatenating chunks"):
            if chunk_path.exists():
                try:
                    all_processed_data.extend(torch.load(chunk_path))
                except Exception as e:
                    # Raise a DataProcessingError for chunk loading failures
                    raise DataProcessingError(
                        message=f"Error loading chunk file {chunk_path} during final concatenation.",
                        details=f"Original error: {e.__class__.__name__}: {e}. Some data might be missing."
                    ) from e
            else:
                # Also raise DataProcessingError if a chunk file is unexpectedly missing
                raise DataProcessingError(
                    message=f"Missing chunk file during final concatenation: {chunk_path}.",
                    details="This indicates an issue during chunk saving or unexpected file deletion."
                )

        self.logger.info(f"Concatenated {len(all_processed_data)} molecules from all chunks.")

        # Only collate and save if there's data to save
        if all_processed_data:
            data: Data
            slices: Dict[str, torch.Tensor]
            data, slices = self.collate(all_processed_data)

            self.logger.debug(f"Type of collated_data before saving: {type((data, slices))}")
            self.logger.debug(f"Collated data is a tuple (data, slices). Data object type: {type(data)}, Slices object type: {type(slices)}")

            # Ensure the processed directory exists before saving the final data.pt
            Path(self.processed_paths[0]).parent.mkdir(parents=True, exist_ok=True)
            torch.save((data, slices), self.processed_paths[0])
            self.logger.info(f"Consolidated processed data saved to {self.processed_paths[0]}")
            self.logger.info("Processing complete.")
        else:
            self.logger.critical("No molecules were successfully processed. 'data.pt' will not be created. Check logs for details on skipped molecules.")
            # Ensure the data.pt file is not created or is empty if no data was processed
            if Path(self.processed_paths[0]).exists():
                os.remove(Path(self.processed_paths[0]))
                self.logger.info(f"Removed empty/incomplete processed file: {self.processed_paths[0]}")

        # Multiprocessing call for cleanup
        self.logger.info("Initiating background cleanup of chunk files using multiprocessing...")
        cleanup_process = multiprocessing.Process(
            target=_delete_directory_in_background,
            args=(str(self.processed_chunk_dir), self.logger.name)
        )
        cleanup_process.daemon = True
        cleanup_process.start()

        self.logger.info("Main processing thread finished. Background cleanup initiated (non-blocking).")
        
