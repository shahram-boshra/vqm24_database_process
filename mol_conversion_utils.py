# mol_conversion_utils.py

"""
Utility functions for converting molecular representations, specifically from SMILES and coordinates
to RDKit molecules, and then to PyTorch Geometric Data objects.

Handles various error conditions and ensures proper data integrity during conversion.
"""
import logging
from typing import Optional
import numpy as np
import torch
from torch_geometric.utils import from_rdmol
from rdkit import Chem
from rdkit.Geometry import Point3D
from torch_geometric.data import Data

from exceptions import MoleculeProcessingError, RDKitConversionError, PyGDataCreationError


logger = logging.getLogger(__name__)


def create_rdkit_mol(
    smiles_string: str,
    coords_np: np.ndarray,
    logger: logging.Logger,
    molecule_index: Optional[int] = None,
    smiles: Optional[str] = None
) -> Chem.Mol:
    """
    Creates an RDKit molecule from a SMILES string and 3D coordinates.

    Args:
        smiles_string (str): SMILES string representing the molecule.
        coords_np (np.ndarray): 3D coordinates for each atom, shape (num_atoms, 3).
        logger (logging.Logger): Logger instance for debugging and error messages.
        molecule_index (Optional[int]): Index of the molecule for error reporting. Defaults to None.
        smiles (Optional[str]): Original SMILES string for error reporting. Defaults to None.

    Returns:
        rdkit.Chem.Mol: The RDKit molecule object with an assigned conformer.

    Raises:
        RDKitConversionError: If molecule creation, sanitization, or coordinate assignment fails critically.
    """
    try:
        # Use sanitize=False to allow parsing of potentially problematic SMILES first
        mol = Chem.MolFromSmiles(smiles_string, sanitize=False)
        if mol is None or mol.GetNumAtoms() == 0:
            raise RDKitConversionError(
                molecule_index=molecule_index,
                smiles=smiles,
                reason="RDKit MolFromSmiles (sanitize=False) resulted in None or empty molecule."
            )
        try:
            # Attempt Kekulization, but proceed if it fails (common for some exotic SMILES)
            Chem.Kekulize(mol)
        except Exception as kek_e:
            logger.debug(
                f"Kekulization failed for SMILES: '{smiles_string}' (Index: {molecule_index}): {kek_e}. Proceeding without it."
            )
            # This is a debug message, not a critical failure leading to an exception
            pass

        mol = Chem.AddHs(mol) # Add hydrogens to the molecule
    except Exception as e: # Catch any exception during initial mol creation/sanitization
        raise RDKitConversionError(
            molecule_index=molecule_index,
            smiles=smiles,
            reason=f"Failed initial RDKit molecule creation or sanitization from SMILES.",
            detail=str(e)
        ) from e

    # After mol creation, check atom count consistency
    if mol.GetNumAtoms() != coords_np.shape[0]:
        raise RDKitConversionError(
            molecule_index=molecule_index,
            smiles=smiles,
            reason=f"Atom count mismatch: RDKit Mol has {mol.GetNumAtoms()} atoms, but coordinates array has {coords_np.shape[0]} atoms.",
            detail="This could lead to incorrect coordinate assignment and model errors."
        )

    conformer = Chem.Conformer(mol.GetNumAtoms())
    try:
        for i in range(mol.GetNumAtoms()):
            x, y, z = coords_np[i]
            conformer.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
        mol.AddConformer(conformer, assignId=True)
    except Exception as e:
        raise RDKitConversionError(
            molecule_index=molecule_index,
            smiles=smiles,
            reason=f"Failed to assign coordinates to RDKit molecule conformer.",
            detail=str(e)
        ) from e

    return mol


def mol_to_pyg_data(
    rdkit_mol: Chem.Mol,
    logger: logging.Logger,
    molecule_index: Optional[int] = None,
    smiles: Optional[str] = None
) -> Data:
    """
    Converts an RDKit molecule object to a PyTorch Geometric Data object.
    Ensures atomic numbers (z) and positions (pos) are correctly set.

    Args:
        rdkit_mol (Chem.Mol): The RDKit molecule object to convert.
        logger (logging.Logger): Logger instance for debugging and error messages.
        molecule_index (Optional[int]): Index of the molecule for error reporting. Defaults to None.
        smiles (Optional[str]): Original SMILES string for error reporting. Defaults to None.

    Returns:
        Data: The PyTorch Geometric Data object containing node features (atomic numbers 'z')
              and 3D positions ('pos').

    Raises:
        PyGDataCreationError: If conversion fails due to missing data, inconsistencies,
                              or inability to assign atomic numbers or positions.
    """
    # Prioritize passed 'smiles' argument for logging/exceptions
    # Fallback to Chem.MolToSmiles if rdkit_mol is valid, otherwise indicate N/A
    smiles_for_log = smiles if smiles is not None else (Chem.MolToSmiles(rdkit_mol) if rdkit_mol else 'N/A')

    # Ensure rdkit_mol is not None before proceeding
    if rdkit_mol is None:
        raise PyGDataCreationError(
            molecule_index=molecule_index,
            smiles=smiles_for_log,
            reason="Input RDKit molecule is None, cannot convert to PyG Data."
        )

    pyg_data: Data = None # Added type hint for initialization
    try:
        pyg_data = from_rdmol(rdkit_mol)
    except Exception as e:
        raise PyGDataCreationError(
            molecule_index=molecule_index,
            smiles=smiles_for_log,
            reason=f"Failed to convert RDKit molecule to basic PyG Data object using from_rdmol.",
            detail=str(e)
        ) from e

    # --- Explicitly set atomic numbers (z) from RDKit molecule ---
    try:
        atomic_numbers = [atom.GetAtomicNum() for atom in rdkit_mol.GetAtoms()]
        if not atomic_numbers: # Check if list is empty
            raise PyGDataCreationError(
                molecule_index=molecule_index,
                smiles=smiles_for_log,
                reason="Failed to extract atomic numbers from RDKit molecule (resulted in empty list)."
            )
        pyg_data.z = torch.tensor(atomic_numbers, dtype=torch.long) # Use torch.long for atomic numbers
    except Exception as e: # Catch any exception during atomic number extraction
        raise PyGDataCreationError(
            molecule_index=molecule_index,
            smiles=smiles_for_log,
            reason="Error extracting or assigning atomic numbers (z) from RDKit molecule.",
            detail=str(e)
        ) from e

    # --- Set positions (pos) from RDKit conformer ---
    try:
        if rdkit_mol.GetNumConformers() > 0:
            conformer = rdkit_mol.GetConformer(0)
            pyg_data.pos = torch.tensor(conformer.GetPositions(), dtype=torch.float)
        else:
            # If no conformer, this is a critical issue for many GNNs requiring 3D coordinates.
            raise PyGDataCreationError(
                molecule_index=molecule_index,
                smiles=smiles_for_log,
                reason="RDKit molecule has no conformer, cannot extract 3D positions (pos)."
                # No detail needed here, as the reason is clear.
            )
    except PyGDataCreationError: # Re-raise if our specific error was already raised
        raise
    except Exception as e: # Catch any other exceptions during position assignment
        raise PyGDataCreationError(
            molecule_index=molecule_index,
            smiles=smiles_for_log,
            reason="Error assigning positions (pos) from RDKit conformer.",
            detail=str(e)
        ) from e

    # After ensuring z is set, let's also ensure num_nodes is consistent with z
    if pyg_data.num_nodes != pyg_data.z.size(0):
        raise PyGDataCreationError(
            molecule_index=molecule_index,
            smiles=smiles_for_log,
            reason=f"Inconsistency detected: pyg_data.num_nodes ({pyg_data.num_nodes}) "
                   f"does not match pyg_data.z.size(0) ({pyg_data.z.size(0)}).",
            detail="This indicates a fundamental mismatch in the graph representation."
        )

    return pyg_data
