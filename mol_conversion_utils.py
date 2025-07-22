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

# AllChem is imported here in the original, but not used in the snippet. Keep for consistency.
from rdkit.Chem import AllChem

from exceptions import MoleculeProcessingError, RDKitConversionError, PyGDataCreationError


logger = logging.getLogger(__name__)


def create_rdkit_mol(
    mol_identifier: str, # Can be SMILES or InChI
    coordinates: np.ndarray, # Renamed for clarity (was coords_np)
    atomic_numbers: np.ndarray, # Explicit atomic numbers for atom creation
    logger: logging.Logger,
    molecule_index: Optional[int] = None,
    mol_id_type: str = 'smiles' # 'smiles' or 'inchi' to control logic
) -> Chem.Mol:
    """
    Creates an RDKit molecule from a SMILES string OR an InChI string,
    along with 3D coordinates and atomic numbers.

    Args:
        mol_identifier (str): SMILES or InChI string representing the molecule.
        coordinates (np.ndarray): 3D coordinates for each atom, shape (num_atoms, 3).
        atomic_numbers (np.ndarray): 1D array of atomic numbers for each atom, shape (num_atoms,).
        logger (logging.Logger): Logger instance for debugging and error messages.
        molecule_index (Optional[int]): Index of the molecule for error reporting. Defaults to None.
        mol_id_type (str): Type of identifier ('smiles' or 'inchi'). Defaults to 'smiles'.

    Returns:
        rdkit.Chem.Mol: The RDKit molecule object with an assigned conformer and stereochemistry
                        derived from the 3D coordinates.

    Raises:
        RDKitConversionError: If molecule creation, sanitization, or coordinate assignment fails critically.
        ValueError: If an unsupported `mol_id_type` is provided.
    """
    context_info = f"Molecule {molecule_index} (Identifier: {mol_identifier})"

    if mol_id_type == 'smiles':
        logger.debug(f"{context_info}: Starting SMILES-based RDKit mol creation.")
        try:
            mol = Chem.MolFromSmiles(mol_identifier, sanitize=False)
            if mol is None or mol.GetNumAtoms() == 0:
                raise RDKitConversionError(
                    molecule_index=molecule_index,
                    inchi=mol_identifier,
                    reason="RDKit MolFromSmiles (sanitize=False) resulted in None or empty molecule."
                )
            try:
                Chem.Kekulize(mol)
            except Exception as kek_e:
                logger.debug(
                    f"Kekulization failed for SMILES: '{mol_identifier}' (Index: {molecule_index}): {kek_e}. Proceeding without it."
                )
                pass

            mol = Chem.AddHs(mol) # Add hydrogens to the molecule
        except Exception as e:
            raise RDKitConversionError(
                molecule_index=molecule_index,
                inchi=mol_identifier,
                reason=f"Failed initial RDKit molecule creation or sanitization from SMILES.",
                detail=str(e)
            ) from e

        if mol.GetNumAtoms() != coordinates.shape[0]:
            raise RDKitConversionError(
                molecule_index=molecule_index,
                inchi=mol_identifier,
                reason=f"Atom count mismatch: RDKit Mol has {mol.GetNumAtoms()} atoms, but coordinates array has {coordinates.shape[0]} atoms.",
                detail="This could lead to incorrect coordinate assignment and model errors."
            )

        conformer = Chem.Conformer(mol.GetNumAtoms())
        try:
            for i in range(mol.GetNumAtoms()):
                x, y, z = coordinates[i]
                conformer.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
            mol.AddConformer(conformer, assignId=True)
        except Exception as e:
            raise RDKitConversionError(
                molecule_index=molecule_index,
                inchi=mol_identifier,
                reason=f"Failed to assign coordinates to RDKit molecule conformer.",
                detail=str(e)
            ) from e

        # --- Start Assign stereochemistry from 3D conformer (SMILES branch) ---
        try:
            # Remove any existing stereochemistry information inferred from the SMILES string.
            # This ensures the 3D coordinates are the single source of truth for stereochemistry.
            # AllChem.RemoveStereochemistry is still valid and useful here.
            AllChem.RemoveStereochemistry(mol)
            # Assign stereochemistry (chiral centers, double bonds) based on the 3D conformer.
            # Using Chem.AssignStereochemistryFrom3D which is the modern and robust RDKit approach.
            Chem.AssignStereochemistryFrom3D(mol)
            logger.debug(f"{context_info}: Assigned stereochemistry from 3D conformer for SMILES mol.")
        except Exception as e:
            # Given that stereochemistry is "crucially vital", we will raise an error here.
            # This will cause the molecule to be skipped by the VQM24Dataset.process() method.
            raise RDKitConversionError(
                molecule_index=molecule_index,
                inchi=mol_identifier, # Use inchi as identifier for logging consistency with InChI branch
                reason=f"Failed to assign stereochemistry from 3D coordinates using Chem.AssignStereochemistryFrom3D for SMILES mol.", 
                detail=str(e)
            ) from e
        # --- END Assign stereochemistry from 3D conformer (SMILES branch) ---

        return mol

    elif mol_id_type == 'inchi':
        logger.debug(f"{context_info}: Starting InChI-based RDKit mol creation.")

        num_atoms_coords = coordinates.shape[0]
        num_atoms_atomic = len(atomic_numbers)

        if num_atoms_coords != num_atoms_atomic:
            raise RDKitConversionError(
                molecule_index=molecule_index,
                inchi=mol_identifier,
                reason="Mismatch between number of atoms in coordinates and 'atoms' data.",
                detail=f"Coordinates have {num_atoms_coords} atoms, 'atoms' data has {num_atoms_atomic}."
            )

        try:
            rdkit_mol = Chem.MolFromInchi(mol_identifier)
            if rdkit_mol is None:
                raise RDKitConversionError(
                    molecule_index=molecule_index,
                    inchi=mol_identifier,
                    reason="Failed to parse InChI string.",
                    detail="Chem.MolFromInchi returned None."
                )
        except Exception as e:
            raise RDKitConversionError(
                molecule_index=molecule_index,
                inchi=mol_identifier,
                reason=f"Error parsing InChI string: {e.__class__.__name__}",
                detail=str(e)
            )

        # Add explicit hydrogens to the RDKit molecule derived from InChI.
        rdkit_mol = Chem.AddHs(rdkit_mol, explicitOnly=False, addCoords=False)

        if rdkit_mol.GetNumAtoms() != num_atoms_coords:
            raise RDKitConversionError(
                molecule_index=molecule_index,
                inchi=mol_identifier,
                reason="Atom count mismatch after AddHs to InChI mol and coordinate array.",
                detail=f"InChI mol has {rdkit_mol.GetNumAtoms()} atoms, coordinates have {num_atoms_coords} atoms."
            )

        conformer = Chem.Conformer(rdkit_mol.GetNumAtoms())
        try:
            for i in range(rdkit_mol.GetNumAtoms()):
                x, y, z = coordinates[i]
                conformer.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
            rdkit_mol.AddConformer(conformer, assignId=True)
        except Exception as e:
            raise RDKitConversionError(
                molecule_index=molecule_index,
                inchi=mol_identifier,
                reason=f"Failed to assign coordinates to RDKit molecule conformer for InChI mol.",
                detail=str(e)
            ) from e

        # --- Start Assign stereochemistry from 3D conformer (InChI branch) ---
        try:
            # Remove any existing stereochemistry information inferred from the InChI string,
            # as we are about to re-assign it from the QM-optimized 3D coordinates.
            # AllChem.RemoveStereochemistry is still valid and useful here.
            AllChem.RemoveStereochemistry(rdkit_mol)
            # Assign stereochemistry (chiral centers, double bonds) based on the 3D conformer.
            # Using Chem.AssignStereochemistryFrom3D which is the modern and robust RDKit approach.
            Chem.AssignStereochemistryFrom3D(rdkit_mol) # <<< Corrected here
            logger.debug(f"{context_info}: Assigned stereochemistry from 3D conformer for InChI mol.")
        except Exception as e:
            # Given that stereochemistry is "crucially vital", we will raise an error here.
            # This will cause the molecule to be skipped by the VQM24Dataset.process() method.
            raise RDKitConversionError(
                molecule_index=molecule_index,
                inchi=mol_identifier,
                reason=f"Failed to assign stereochemistry from 3D coordinates using Chem.AssignStereochemistryFrom3D for InChI mol.", # <<< Corrected here
                detail=str(e)
            ) from e
        # --- END Assign stereochemistry from 3D conformer (InChI branch) ---

        # 6. Final validation: Generate InChI from the constructed RDKit Mol and compare to input.
        try:
            generated_inchi = Chem.MolToInchi(rdkit_mol)
            if generated_inchi != mol_identifier:
                # Log a warning, but don't raise a hard error for InChI mismatch IF stereochemistry
                # was explicitly re-assigned from conformer. The goal is to use the 3D structure.
                # However, if this happens consistently, it points to a deeper issue with the
                # consistency between input InChI string and QM geometry.
                logger.warning(
                    f"{context_info}: Generated InChI from RDKit Mol (after 3D stereochemistry assignment) "
                    f"does NOT match input InChI. This might be expected if input InChI did not fully "
                    f"capture 3D stereochemistry or if 3D coordinates conflict with canonical InChI. "
                    f"Input InChI: {mol_identifier}, Generated InChI: {generated_inchi}"
                )
        except Exception as inchi_err:
            # Still raise an error if we can't even generate the InChI for validation
            raise RDKitConversionError(
                molecule_index=molecule_index,
                inchi=mol_identifier,
                reason="Failed to generate InChI for validation after RDKit mol creation.",
                detail=str(inchi_err)
            )

        return rdkit_mol

    else:
        raise ValueError(f"Unsupported mol_id_type: {mol_id_type}. Must be 'smiles' or 'inchi'.")


def mol_to_pyg_data(
    rdkit_mol: Chem.Mol,
    logger: logging.Logger,
    molecule_index: Optional[int] = None,
    inchi: Optional[str] = None
) -> Data:
    """
    Converts an RDKit molecule object to a PyTorch Geometric Data object.
    Ensures atomic numbers (z) and positions (pos) are correctly set.

    Args:
        rdkit_mol (Chem.Mol): The RDKit molecule object to convert.
        logger (logging.Logger): Logger instance for debugging and error messages.
        molecule_index (Optional[int]): Index of the molecule for error reporting. Defaults to None.
        inchi (Optional[str]): The molecular identifier (SMILES or InChI) for error reporting. Defaults to None.

    Returns:
        Data: The PyTorch Geometric Data object containing node features (atomic numbers 'z')
              and 3D positions ('pos').

    Raises:
        PyGDataCreationError: If conversion fails due to missing data, inconsistencies,
                              or inability to assign atomic numbers or positions.
    """
    if inchi is not None:
        identifier_for_log = inchi
    elif rdkit_mol:
        try:
            identifier_for_log = Chem.MolToInchi(rdkit_mol)
        except Exception:
            try:
                identifier_for_log = Chem.MolToSmiles(rdkit_mol)
            except Exception:
                identifier_for_log = 'N/A (RDKit conversion failed)'
    else:
        identifier_for_log = 'N/A (unknown mol)'


    if rdkit_mol is None:
        raise PyGDataCreationError(
            molecule_index=molecule_index,
            inchi=identifier_for_log,
            reason="Input RDKit molecule is None, cannot convert to PyG Data."
        )

    pyg_data: Data = None
    try:
        pyg_data = from_rdmol(rdkit_mol)
    except Exception as e:
        raise PyGDataCreationError(
            molecule_index=molecule_index,
            inchi=identifier_for_log,
            reason=f"Failed to convert RDKit molecule to basic PyG Data object using from_rdmol.",
            detail=str(e)
        ) from e

    # --- Explicitly set atomic numbers (z) from RDKit molecule ---
    try:
        atomic_numbers_list = [atom.GetAtomicNum() for atom in rdkit_mol.GetAtoms()]
        if not atomic_numbers_list:
            raise PyGDataCreationError(
                molecule_index=molecule_index,
                inchi=identifier_for_log,
                reason="Failed to extract atomic numbers from RDKit molecule (resulted in empty list)."
            )
        pyg_data.z = torch.tensor(atomic_numbers_list, dtype=torch.long)
    except Exception as e:
        raise PyGDataCreationError(
            molecule_index=molecule_index,
            inchi=identifier_for_log,
            reason="Error extracting or assigning atomic numbers (z) from RDKit molecule.",
            detail=str(e)
        ) from e

    # --- Set positions (pos) from RDKit conformer ---
    try:
        if rdkit_mol.GetNumConformers() > 0:
            conformer = rdkit_mol.GetConformer(0)
            pyg_data.pos = torch.tensor(conformer.GetPositions(), dtype=torch.float)
        else:
            raise PyGDataCreationError(
                molecule_index=molecule_index,
                inchi=identifier_for_log,
                reason="RDKit molecule has no conformer, cannot extract 3D positions (pos)."
            )
    except PyGDataCreationError:
        raise
    except Exception as e:
        raise PyGDataCreationError(
            molecule_index=molecule_index,
            inchi=identifier_for_log,
            reason="Error assigning positions (pos) from RDKit conformer.",
            detail=str(e)
        ) from e

    if pyg_data.num_nodes != pyg_data.z.size(0):
        raise PyGDataCreationError(
            molecule_index=molecule_index,
            inchi=identifier_for_log,
            reason=f"Inconsistency detected: pyg_data.num_nodes ({pyg_data.num_nodes}) "
                    f"does not match pyg_data.z.size(0) ({pyg_data.z.size(0)}).",
            detail="This indicates a fundamental mismatch in the graph representation."
        )

    return pyg_data
