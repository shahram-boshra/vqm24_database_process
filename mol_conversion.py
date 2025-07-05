# mol_conversion.py

"""
Module for converting raw molecular data (SMILES, coordinates) into RDKit
molecules and then into PyTorch Geometric Data objects, enriching them with
additional properties.

Handles various exceptions during the conversion process for robust data pipeline.
"""
import logging
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem # AllChem is imported but not directly used in the provided snippet
from torch_geometric.data import Data
from typing import Dict, Optional, Union, Any

from exceptions import (
    MoleculeProcessingError,
    RDKitConversionError,
    PyGDataCreationError,
    PropertyEnrichmentError,
    MoleculeFilterRejectedError
)

from mol_conversion_utils import create_rdkit_mol, mol_to_pyg_data
from property_enrichment import enrich_pyg_data_with_properties
from mol_structural_features import add_structural_features


logger = logging.getLogger(__name__)


class MoleculeDataConverter:
    """
    Converts raw molecular data into a PyTorch Geometric Data object.

    This class orchestrates the conversion process, including RDKit molecule
    creation, PyTorch Geometric data object generation, and property enrichment.
    It robustly handles various conversion-related exceptions.
    """
    def __init__(self,
                 logger: logging.Logger,
                 structural_features_config: Dict[str, Any] = None):

        """
        Initializes the MoleculeDataConverter with a logger instance.

        Args:
            logger (logging.Logger): The logger instance to use for
                                     logging messages and errors.
        """
        self.logger = logger
        self.structural_features_config = structural_features_config if structural_features_config is not None else {}


    def convert(self,
                molecule_index: int,
                raw_properties_dict: Dict[str, Union[str, np.ndarray]],
                config: Dict) -> Optional[Data]:
        """
        Converts raw molecular data into a PyTorch Geometric Data object.

        This method orchestrates the full conversion pipeline:
        1. Extracts molecular identifier (InChI/SMILES), coordinates, and atomic numbers from `raw_properties_dict`.
        2. Creates an RDKit molecule using `create_rdkit_mol`.
        3. Converts the RDKit molecule to a PyTorch Geometric `Data` object
           using `mol_to_pyg_data`.
        4. Enriches the PyG `Data` object with additional properties using
           `enrich_pyg_data_with_properties`.
        5. Attaches original identifier (SMILES/InChI) and molecule index to the `Data` object.
        It handles various custom exceptions by logging warnings and returning `None`,
        and logs critical errors for unhandled exceptions.

        Args:
            molecule_index (int): The original index of the molecule in the dataset.
            raw_properties_dict (Dict[str, Union[str, np.ndarray]]): A dictionary
                containing raw molecular data, expected to have 'inchi' or 'graphs' (SMILES),
                'coordinates', and 'atoms' keys, along with other properties.
            config (Dict): A configuration dictionary, primarily used for
                property enrichment settings.

        Returns:
            Optional[Data]: A PyTorch Geometric `Data` object if conversion is
                            successful, otherwise `None` if an expected error occurs
                            or the molecule is filtered.
        """
        self.logger.debug(f"Molecule {molecule_index}: Full config received by convert method: {config}")

        # Capture the current molecule's identifier and index for all error contexts
        current_mol_index: int = molecule_index
        current_mol_identifier: str = "N/A (unknown)"
        mol_id_type: str = 'unknown' # Will be 'smiles' or 'inchi'

        try:
            # Retrieve identifier (prefer InChI), coordinates, and atomic numbers
            mol_identifier: Optional[str] = raw_properties_dict.get('inchi')
            if mol_identifier is not None:
                mol_id_type = 'inchi'
            else:
                mol_identifier = raw_properties_dict.get('graphs') # Fallback to SMILES
                if mol_identifier is not None:
                    mol_id_type = 'smiles'

            coordinates: Optional[np.ndarray] = raw_properties_dict.get('coordinates')
            atomic_numbers: Optional[np.ndarray] = raw_properties_dict.get('atoms') # New: atomic numbers

            # Update current_mol_identifier once it's retrieved, before any potential early exits
            if mol_identifier is not None:
                current_mol_identifier = str(mol_identifier)
            # If mol_identifier is None, current_mol_identifier remains "N/A (unknown)" which is appropriate for the error.

            # Validate essential inputs
            if mol_identifier is None:
                raise RDKitConversionError(
                    molecule_index=current_mol_index,
                    inchi="N/A (missing from raw data)", # Changed to inchi
                    reason="Molecular identifier (InChI or SMILES) not found in raw data.",
                    detail="Cannot create RDKit molecule without a molecular identifier ('inchi' or 'graphs' key)."
                )
            if coordinates is None:
                raise RDKitConversionError(
                    molecule_index=current_mol_index,
                    inchi=current_mol_identifier, # Changed to inchi
                    reason="Coordinates not found in raw data.",
                    detail="Cannot create RDKit molecule without coordinates."
                )
            if atomic_numbers is None: # NEW CHECK
                raise RDKitConversionError(
                    molecule_index=current_mol_index,
                    inchi=current_mol_identifier, # Changed to inchi
                    reason="Atomic numbers ('atoms' key) not found in raw data.",
                    detail="Cannot create RDKit molecule for InChI/QM data without explicit atomic numbers."
                )

            # Ensure mol_identifier is a string (though 'get' should return str or None)
            mol_identifier = str(mol_identifier)
            current_mol_identifier = mol_identifier # Ensure this is always up-to-date for logs/exceptions

            # Pass molecule_index, identifier, coordinates, atomic_numbers, and type to create_rdkit_mol
            # create_rdkit_mol will now raise RDKitConversionError on failure
            rdkit_mol: Chem.Mol = create_rdkit_mol(
                mol_identifier=mol_identifier,
                coordinates=coordinates,
                atomic_numbers=atomic_numbers,
                logger=self.logger,
                molecule_index=current_mol_index,
                mol_id_type=mol_id_type # Keep this
            )

            # Pass molecule_index and identifier to the utility function
            # mol_to_pyg_data will now raise PyGDataCreationError on failure
            pyg_data: Data = mol_to_pyg_data(rdkit_mol, self.logger,
                                             molecule_index=current_mol_index, inchi=current_mol_identifier) # Changed to inchi

            # --- START NEW CODE BLOCK: Add structural features ---
            # Get the configuration for structural features from the main config
            structural_features_config = self.structural_features_config

            if structural_features_config: # Only call if configuration for features exists
                self.logger.debug(f"Adding structural features for mol index {current_mol_index}, identifier '{current_mol_identifier}'")
                pyg_data = add_structural_features(
                    rdkit_mol=rdkit_mol,
                    pyg_data=pyg_data,
                    feature_config=structural_features_config,
                    logger=self.logger, # Added explicit keyword
                    molecule_index=current_mol_index,
                    inchi=current_mol_identifier # Changed to inchi
                )
            else:
                self.logger.info(f"No structural_features configuration found for mol index {current_mol_index}, identifier '{current_mol_identifier}'. Skipping feature addition.")
                # Ensure x and edge_attr are set to None if no features are added
                pyg_data.x = None
                pyg_data.edge_attr = None
            # --- END NEW CODE BLOCK ---

            # Attach identifier and original_mol_idx to pyg_data for better debugging downstream
            # Store the primary identifier in 'inchi' if it was InChI, otherwise in 'smiles'
            if mol_id_type == 'inchi':
                pyg_data.inchi = current_mol_identifier
                pyg_data.smiles = None # Clear smiles if it was an InChI
            else: # mol_id_type is 'smiles'
                pyg_data.smiles = current_mol_identifier
                pyg_data.inchi = None # Clear inchi if it was a SMILES

            pyg_data.original_mol_idx = current_mol_index


            # Pass raw_properties_dict for property enrichment
            pyg_data = enrich_pyg_data_with_properties(
                pyg_data,
                current_mol_index,
                raw_properties_dict,
                inchi_identifier=current_mol_identifier, # Changed to inchi_identifier
                logger=self.logger, # Added explicit keyword
                data_config=config
            )

            return pyg_data


        # Catch specific custom exceptions first, from most specific to more general
        except MoleculeFilterRejectedError as e:
            # This block catches expected molecule rejections due to configured filters.
            # These are typically not "errors" in a critical sense but part of data selection.
            self.logger.info(f"Molecule filtered: {e}") # Log at INFO level for expected filtering
            return None
        except RDKitConversionError as e:
            # Catch errors specifically from RDKit molecule creation/processing
            self.logger.warning(f"RDKit conversion error: {e}") # e.__str__() is already informative
            return None
        except PyGDataCreationError as e:
            # Catch errors specifically from PyTorch Geometric Data object creation
            self.logger.warning(f"PyG Data creation error: {e}") # e.__str__() is already informative
            return None
        except PropertyEnrichmentError as e:
            # Catch errors specifically from property enrichment
            self.logger.warning(f"Property enrichment error: {e}") # e.__str__() is already informative
            return None
        except MoleculeProcessingError as e:
            # This general MoleculeProcessingError catches any other general processing issues
            # that might be raised by utility functions or implicitly (e.g., if a new utility raises this base class).
            self.logger.warning(f"General molecule processing skipped: {e}")
            return None
        except Exception as e:
            # This is the catch-all for any *unexpected* errors that were not handled
            # by our specific custom exceptions. This is critical for identifying bugs.
            self.logger.critical(
                f"UNHANDLED CRITICAL ERROR converting molecule {current_mol_index} (Identifier: {current_mol_identifier}): {e.__class__.__name__} - {e}",
                exc_info=True # Always log traceback for unhandled errors
            )
            return None
