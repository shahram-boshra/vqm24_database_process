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
from rdkit.Chem import AllChem
from torch_geometric.data import Data
from typing import Dict, Optional, Union


from exceptions import (
    MoleculeProcessingError,
    RDKitConversionError,
    PyGDataCreationError,
    PropertyEnrichmentError,
    MoleculeFilterRejectedError
)


from mol_conversion_utils import create_rdkit_mol, mol_to_pyg_data
from property_enrichment import enrich_pyg_data_with_properties


logger = logging.getLogger(__name__)


class MoleculeDataConverter:
    """
    Converts raw molecular data into a PyTorch Geometric Data object.

    This class orchestrates the conversion process, including RDKit molecule
    creation, PyTorch Geometric data object generation, and property enrichment.
    It robustly handles various conversion-related exceptions.
    """
    def __init__(self, logger: logging.Logger):
        """
        Initializes the MoleculeDataConverter with a logger instance.

        Args:
            logger (logging.Logger): The logger instance to use for
                                     logging messages and errors.
        """
        self.logger = logger

    def convert(self,
                molecule_index: int,
                raw_properties_dict: Dict[str, Union[str, np.ndarray]],
                config: Dict) -> Optional[Data]:
        """
        Converts raw molecular data into a PyTorch Geometric Data object.

        This method orchestrates the full conversion pipeline:
        1. Extracts SMILES and coordinates from `raw_properties_dict`.
        2. Creates an RDKit molecule using `create_rdkit_mol`.
        3. Converts the RDKit molecule to a PyTorch Geometric `Data` object
           using `mol_to_pyg_data`.
        4. Enriches the PyG `Data` object with additional properties using
           `enrich_pyg_data_with_properties`.
        5. Attaches original SMILES and molecule index to the `Data` object.
        It handles various custom exceptions by logging warnings and returning `None`,
        and logs critical errors for unhandled exceptions.

        Args:
            molecule_index (int): The original index of the molecule in the dataset.
            raw_properties_dict (Dict[str, Union[str, np.ndarray]]): A dictionary
                containing raw molecular data, expected to have 'graphs' (SMILES)
                and 'coordinates' keys, along with other properties.
            config (Dict): A configuration dictionary, primarily used for
                property enrichment settings.

        Returns:
            Optional[Data]: A PyTorch Geometric `Data` object if conversion is
                            successful, otherwise `None` if an expected error occurs
                            or the molecule is filtered.
        """
        # Capture the current molecule's SMILES and index for all error contexts
        current_mol_index: int = molecule_index
        current_smiles: str = "N/A (unknown)"

        try:
            # Retrieve SMILES and coordinates from the raw_properties_dict
            smiles_str: Optional[str] = raw_properties_dict.get('graphs')
            coordinates: Optional[np.ndarray] = raw_properties_dict.get('coordinates')

            # Update current_smiles once it's retrieved, before any potential early exits
            if smiles_str is not None:
                current_smiles = str(smiles_str)
            # If smiles_str is None, current_smiles remains "N/A (unknown)" which is appropriate for the error.

            if smiles_str is None:
                # Raise specific RDKitConversionError for missing SMILES
                raise RDKitConversionError(
                    molecule_index=current_mol_index,
                    smiles="N/A (missing from raw data)", # Make this explicit for the exception
                    reason="SMILES string ('graphs' key) not found in raw data.",
                    detail="Cannot create RDKit molecule without SMILES."
                )
            if coordinates is None:
                # Raise specific RDKitConversionError for missing coordinates
                raise RDKitConversionError(
                    molecule_index=current_mol_index,
                    smiles=current_smiles, # Use the potentially updated current_smiles
                    reason="Coordinates not found in raw data.",
                    detail="Cannot create RDKit molecule without coordinates."
                )

            # Ensure smiles_str is actually a string (though 'get' should return str or None)
            smiles_str = str(smiles_str)
            current_smiles = smiles_str # Ensure this is always up-to-date for logs/exceptions


            # Pass molecule_index and smiles to the utility function
            # create_rdkit_mol will now raise RDKitConversionError on failure
            rdkit_mol: Chem.Mol = create_rdkit_mol(smiles_str, coordinates, self.logger,
                                                   molecule_index=current_mol_index, smiles=current_smiles)


            # Pass molecule_index and smiles to the utility function
            # mol_to_pyg_data will now raise PyGDataCreationError on failure
            pyg_data: Data = mol_to_pyg_data(rdkit_mol, self.logger,
                                             molecule_index=current_mol_index, smiles=current_smiles)


            # Attach smiles and original_mol_idx to pyg_data for better debugging downstream
            pyg_data.smiles = current_smiles
            pyg_data.original_mol_idx = current_mol_index

            # Pass raw_properties_dict for property enrichment
            pyg_data = enrich_pyg_data_with_properties(
                pyg_data,
                current_mol_index,
                raw_properties_dict,
                current_smiles,
                self.logger,
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
                f"UNHANDLED CRITICAL ERROR converting molecule {current_mol_index} (SMILES: {current_smiles}): {e.__class__.__name__} - {e}",
                exc_info=True # Always log traceback for unhandled errors
            )
            return None
        
