# mol_tructural_features.py

import logging
import torch
from rdkit import Chem
from rdkit.Chem import HybridizationType, BondType
from torch_geometric.data import Data
from typing import Optional, Dict, List, Tuple

from exceptions import MoleculeProcessingError, StructuralFeatureError, PyGDataCreationError

logger = logging.getLogger(__name__)

# --- Helper for One-Hot Encoding (common utility) ---
def _one_hot_encoding(value, choices: List) -> List[int]:
    """
    Generates a one-hot encoding for a given value based on a list of choices.

    Args:
        value: The value to encode.
        choices (List): A list of possible values that `value` can take.

    Returns:
        List[int]: A list of integers (0s and 1s) representing the one-hot encoding.
                   Returns all zeros if the value is not found in choices.
    """
    encoding = [0] * len(choices)
    try:
        idx = choices.index(value)
        encoding[idx] = 1
    except ValueError:
        # Value not in choices, all zeros. This can be desirable for 'unknown' or 'other'
        pass
    return encoding


# --- Atom Feature Calculation Functions ---
def _get_atom_degree(atom: Chem.Atom) -> int:
    """
    Returns the number of directly bonded neighbors (excluding implicit/explicit Hs).

    Args:
        atom (Chem.Atom): The RDKit atom object.

    Returns:
        int: The degree of the atom.
    """
    return atom.GetDegree()

def _get_atom_total_degree(atom: Chem.Atom) -> int:
    """
    Returns the total number of bonds to an atom (including implicit/explicit Hs).

    Args:
        atom (Chem.Atom): The RDKit atom object.

    Returns:
        int: The total degree of the atom.
    """
    return atom.GetTotalDegree()

def _get_atom_hybridization_feature(atom: Chem.Atom) -> List[int]:
    """
    Returns a one-hot encoding of the atom's hybridization state.

    Args:
        atom (Chem.Atom): The RDKit atom object.

    Returns:
        List[int]: A one-hot encoded list representing the hybridization type.
    """
    # Common RDKit Hybridization types
    hybridization_choices = [
        HybridizationType.S,
        HybridizationType.SP,
        HybridizationType.SP2,
        HybridizationType.SP3,
        HybridizationType.SP3D,
        HybridizationType.SP3D2,
        HybridizationType.UNSPECIFIED # Handle cases where hybridization isn't clearly defined
    ]
    return _one_hot_encoding(atom.GetHybridization(), hybridization_choices)

def _get_atom_total_valence(atom: Chem.Atom) -> int:
    """
    Returns the total valence of the atom.

    Args:
        atom (Chem.Atom): The RDKit atom object.

    Returns:
        int: The total valence.
    """
    return atom.GetTotalValence()

def _is_atom_aromatic(atom: Chem.Atom) -> int:
    """
    Checks if the atom is aromatic.

    Args:
        atom (Chem.Atom): The RDKit atom object.

    Returns:
        int: 1 if the atom is aromatic, 0 otherwise.
    """
    return int(atom.GetIsAromatic())

def _is_atom_in_ring(atom: Chem.Atom) -> int:
    """
    Checks if the atom is part of any ring.

    Args:
        atom (Chem.Atom): The RDKit atom object.

    Returns:
        int: 1 if the atom is in a ring, 0 otherwise.
    """
    return int(atom.IsInRing())


# --- Bond Feature Calculation Functions ---
def _get_bond_type_feature(bond: Chem.Bond) -> List[int]:
    """
    Returns a one-hot encoding of the bond's type.

    Args:
        bond (Chem.Bond): The RDKit bond object.

    Returns:
        List[int]: A one-hot encoded list representing the bond type.
    """
    # Common RDKit Bond types
    bond_type_choices = [
        BondType.SINGLE,
        BondType.DOUBLE,
        BondType.TRIPLE,
        BondType.AROMATIC # Aromatic bonds are often treated as a separate type
    ]
    return _one_hot_encoding(bond.GetBondType(), bond_type_choices)

def _is_bond_conjugated(bond: Chem.Bond) -> int:
    """
    Checks if the bond is conjugated.

    Args:
        bond (Chem.Bond): The RDKit bond object.

    Returns:
        int: 1 if the bond is conjugated, 0 otherwise.
    """
    return int(bond.GetIsConjugated())

def _is_bond_aromatic(bond: Chem.Bond) -> int:
    """
    Checks if the bond is aromatic.

    Args:
        bond (Chem.Bond): The RDKit bond object.

    Returns:
        int: 1 if the bond is aromatic, 0 otherwise.
    """
    return int(bond.GetIsAromatic())

def _is_bond_in_any_ring(bond: Chem.Bond) -> int:
    """
    Checks if the bond is part of any ring.

    Args:
        bond (Chem.Bond): The RDKit bond object.

    Returns:
        int: 1 if the bond is in a ring, 0 otherwise.
    """
    return int(bond.IsInRing())


# --- Aggregation functions for atom and bond features ---
def _calculate_atom_features_tensor(rdkit_mol: Chem.Mol, selected_features: List[str],
                                     molecule_index: Optional[int] = None, inchi: Optional[str] = None) -> torch.Tensor: 
    """
    Calculates atom features based on a list of selected feature names and returns them as a PyTorch tensor.

    Args:
        rdkit_mol (Chem.Mol): The RDKit molecule object.
        selected_features (List[str]): A list of string names for the atom features to calculate (e.g., "degree", "hybridization").
        molecule_index (Optional[int]): The index of the molecule in the dataset, for error context. Defaults to None.
        inchi (Optional[str]): The InChI string of the molecule, for error context. Defaults to None. 

    Returns:
        torch.Tensor: A tensor of atom features, where each row corresponds to an atom
                      and columns are the concatenated features. Shape: (num_atoms, feature_dim).
                      Returns an empty tensor if no atoms or no features are selected.

    Raises:
        StructuralFeatureError: If an unsupported atom feature is requested,
                                if an error occurs during an RDKit atom feature calculation,
                                or if the final tensor conversion fails.
    """
    atom_feature_map = {
        "degree": _get_atom_degree,
        "total_degree": _get_atom_total_degree,
        "hybridization": _get_atom_hybridization_feature,
        "total_valence": _get_atom_total_valence,
        "is_aromatic": _is_atom_aromatic,
        "is_in_ring": _is_atom_in_ring,
    }

    features_list = []
    for i, atom in enumerate(rdkit_mol.GetAtoms()):
        atom_feature_vector = []
        for feature_name in selected_features:
            if feature_name in atom_feature_map:
                try:
                    feature_value = atom_feature_map[feature_name](atom)
                    if isinstance(feature_value, list):
                        atom_feature_vector.extend(feature_value)
                    else:
                        atom_feature_vector.append(feature_value)
                except Exception as e:
                    # Catch errors from RDKit atom methods
                    raise StructuralFeatureError(
                        message=f"Error calculating '{feature_name}' for atom {i}.",
                        molecule_index=molecule_index,
                        inchi=inchi, 
                        feature_type="atom",
                        feature_name=feature_name,
                        reason=f"Failed to retrieve feature value for atom {i}.",
                        detail=str(e)
                    ) from e
            else:
                # If an unknown feature is requested, we raise an error.
                raise StructuralFeatureError(
                    message=f"Unsupported atom feature requested: '{feature_name}'.",
                    molecule_index=molecule_index,
                    inchi=inchi, 
                    feature_type="atom",
                    feature_name=feature_name,
                    reason="Invalid feature configuration.",
                    detail="Ensure all atom features specified in the configuration are recognized."
                )
        features_list.append(atom_feature_vector)

    if not features_list:
        # Handle case with no atoms or no selected features.
        return torch.empty(0, 0, dtype=torch.float)

    # Ensure all atom feature vectors have the same length (important for torch.tensor)
    max_len = max(len(vec) for vec in features_list)
    padded_features_list = [vec + [0] * (max_len - len(vec)) for vec in features_list]

    try:
        return torch.tensor(padded_features_list, dtype=torch.float)
    except Exception as e:
        raise StructuralFeatureError(
            message="Failed to convert atom features to a PyTorch tensor.",
            molecule_index=molecule_index,
            inchi=inchi, 
            feature_type="atom",
            reason="Inconsistent feature vector lengths or invalid data.",
            detail=str(e)
        ) from e


def _calculate_bond_features_tensor(rdkit_mol: Chem.Mol, pyg_edge_index: torch.Tensor, selected_features: List[str],
                                     molecule_index: Optional[int] = None, inchi: Optional[str] = None) -> torch.Tensor:
    """
    Calculates bond features based on a list of selected feature names and returns them as a PyTorch tensor.
    Ensures features align with the provided PyTorch Geometric edge_index (bidirectional).

    Args:
        rdkit_mol (Chem.Mol): The RDKit molecule object.
        pyg_edge_index (torch.Tensor): The PyTorch Geometric edge_index tensor (shape [2, num_edges]),
                                         representing graph connectivity.
        selected_features (List[str]): A list of string names for the bond features to calculate (e.g., "bond_type", "is_aromatic").
        molecule_index (Optional[int]): The index of the molecule in the dataset, for error context. Defaults to None.
        inchi (Optional[str]): The InChI string of the molecule, for error context. Defaults to None. 

    Returns:
        torch.Tensor: A tensor of bond features, where each row corresponds to an edge in `pyg_edge_index`
                      and columns are the concatenated features. Shape: (num_edges, feature_dim).
                      Returns an empty tensor if no edges or no features are selected.
                      Assigns zeros to features for PyG edges that do not correspond to an explicit RDKit bond.

    Raises:
        StructuralFeatureError: If an unsupported bond feature is requested,
                                if an error occurs during an RDKit bond feature calculation,
                                if the dummy bond creation for feature length fails,
                                or if the final tensor conversion fails.
    """
    bond_feature_map = {
        "bond_type": _get_bond_type_feature,
        "is_conjugated": _is_bond_conjugated,
        "is_aromatic": _is_bond_aromatic,
        "is_in_any_ring": _is_bond_in_any_ring,
    }

    # Determine the length of a single bond feature vector for padding
    single_bond_feature_length = 0
    if selected_features: # Only attempt if features are actually requested
        try:
            # A dummy bond to get feature length. Ensure its creation is robust.
            dummy_mol = Chem.MolFromSmiles("C-C")
            if dummy_mol is None or dummy_mol.GetNumBonds() == 0:
                raise ValueError("Could not create dummy molecule for bond feature length determination.")
            dummy_bond = dummy_mol.GetBondWithIdx(0)
            
            for feature_name in selected_features:
                if feature_name in bond_feature_map:
                    val = bond_feature_map[feature_name](dummy_bond)
                    single_bond_feature_length += len(val) if isinstance(val, list) else 1
                else:
                    # Raise error if an unknown bond feature is requested at config time
                    raise StructuralFeatureError(
                        message=f"Unsupported bond feature requested: '{feature_name}'.",
                        molecule_index=molecule_index,
                        inchi=inchi, 
                        feature_type="bond",
                        feature_name=feature_name,
                        reason="Invalid feature configuration.",
                        detail="Ensure all bond features specified in the configuration are recognized."
                    )
            
            if single_bond_feature_length == 0:
                logger.warning(f"[{molecule_index}, '{inchi}'] No valid bond features selected or they have 0 length. Defaulting to 1 for dummy feature length.") 
                single_bond_feature_length = 1 

        except Exception as e:
            raise StructuralFeatureError(
                message="Failed to determine expected bond feature vector length.",
                molecule_index=molecule_index,
                inchi=inchi, 
                feature_type="bond",
                reason="Error during dummy bond processing or feature map lookup.",
                detail=str(e)
            ) from e
    else: # No bond features selected
        single_bond_feature_length = 0 # No features means 0 length

    # Create a mapping from RDKit bond (represented by sorted atom indices) to its features
    rdkit_bond_features_dict: Dict[Tuple[int, int], List[float]] = {}
    
    for i, bond in enumerate(rdkit_mol.GetBonds()):
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        
        bond_feature_vector = []
        for feature_name in selected_features:
            if feature_name in bond_feature_map:
                try:
                    feature_value = bond_feature_map[feature_name](bond)
                    if isinstance(feature_value, list):
                        bond_feature_vector.extend(feature_value)
                    else:
                        bond_feature_vector.append(feature_value)
                except Exception as e:
                    # Catch errors from RDKit bond methods
                    raise StructuralFeatureError(
                        message=f"Error calculating '{feature_name}' for bond between atoms {u}-{v}.",
                        molecule_index=molecule_index,
                        inchi=inchi, 
                        feature_type="bond",
                        feature_name=feature_name,
                        reason=f"Failed to retrieve feature value for bond {i}.",
                        detail=str(e)
                    ) from e
        
        # Ensure the feature vector has consistent length, even if some features were skipped
        bond_feature_vector_padded = bond_feature_vector + [0] * (single_bond_feature_length - len(bond_feature_vector))
        
        # Store features for both directions for easy lookup
        rdkit_bond_features_dict[(u, v)] = bond_feature_vector_padded
        rdkit_bond_features_dict[(v, u)] = bond_feature_vector_padded # For reverse direction in PyG

    # Now, construct the PyG edge_attr tensor using pyg_edge_index
    num_edges = pyg_edge_index.size(1)
    
    if num_edges == 0:
        return torch.empty(0, single_bond_feature_length, dtype=torch.float)

    edge_features_list = []
    for i in range(num_edges):
        u, v = pyg_edge_index[0, i].item(), pyg_edge_index[1, i].item()
        features = rdkit_bond_features_dict.get((u, v))
        if features is None:
            logger.warning(f"[{molecule_index}, '{inchi}'] No RDKit bond found for PyG edge ({u}, {v}). Assigning zeros to features.") 
            edge_features_list.append([0] * single_bond_feature_length)
        else:
            edge_features_list.append(features)

    try:
        return torch.tensor(edge_features_list, dtype=torch.float)
    except Exception as e:
        raise StructuralFeatureError(
            message="Failed to convert bond features to a PyTorch tensor.",
            molecule_index=molecule_index,
            inchi=inchi, 
            feature_type="bond",
            reason="Inconsistent feature vector lengths or invalid data.",
            detail=str(e)
        ) from e


# --- Main function to add structural features to PyG Data object ---
def add_structural_features(
    rdkit_mol: Chem.Mol,
    pyg_data: Data,
    feature_config: Dict,
    logger: logging.Logger,
    molecule_index: Optional[int] = None,
    inchi: Optional[str] = None 
) -> Data:
    """
    Adds atom-level (pyg_data.x) and bond-level (pyg_data.edge_attr) structural features
    to a PyTorch Geometric Data object based on a provided feature configuration.

    Args:
        rdkit_mol (Chem.Mol): The RDKit molecule object from which to extract features.
        pyg_data (Data): The PyTorch Geometric Data object to enrich with features.
                         Must contain 'edge_index' if bond features are requested.
        feature_config (Dict): A dictionary specifying which features to add.
                               Expected keys: "atom" and "bond", each with a list of feature names (str).
                               Example: {"atom": ["degree", "hybridization"], "bond": ["bond_type"]}
        logger (logging.Logger): A logger instance for logging informational messages and warnings.
        molecule_index (Optional[int]): The index of the molecule in the dataset. Used for detailed error reporting.
                                        Defaults to None.
        inchi (Optional[str]): The InChI string of the molecule. Used for detailed error reporting. 
                               Defaults to None.

    Returns:
        Data: The modified PyTorch Geometric Data object with 'x' (atom features)
              and 'edge_attr' (bond features) tensors populated.
              'x' or 'edge_attr' will be None if no corresponding features are configured or applicable.

    Raises:
        MoleculeProcessingError: If the input `rdkit_mol` is None.
        PyGDataCreationError: If the input `pyg_data` is None.
        StructuralFeatureError: If any error occurs during the calculation, encoding,
                                or assignment of atom or bond features (e.g., unsupported feature names,
                                RDKit errors during feature extraction, tensor conversion issues).
                                This acts as a wrapper for more specific issues from helper functions.
    """
    log_prefix = f"[Mol Index: {molecule_index}, InChI: '{inchi}']" if molecule_index is not None else "" 

    # Validate inputs
    if rdkit_mol is None:
        raise MoleculeProcessingError(
            message="RDKit molecule object is None.",
            molecule_index=molecule_index,
            inchi=inchi, 
            reason="Input 'rdkit_mol' is invalid.",
            detail="Cannot extract structural features from a non-existent molecule."
        )
    if pyg_data is None:
        raise PyGDataCreationError(
            message="PyTorch Geometric Data object is None.",
            molecule_index=molecule_index,
            inchi=inchi, 
            reason="Input 'pyg_data' is invalid.",
            detail="Cannot add structural features to a non-existent PyG Data object."
        )


    selected_atom_features = feature_config.get("atom", [])
    selected_bond_features = feature_config.get("bond", [])

    try:
        # Calculate and assign atom features (pyg_data.x)
        if selected_atom_features:
            if rdkit_mol.GetNumAtoms() == 0:
                logger.warning(f"{log_prefix} RDKit molecule has no atoms. Atom features will be empty.")
                try:
                    dummy_mol_for_dim = Chem.MolFromSmiles("C")
                    if dummy_mol_for_dim and dummy_mol_for_dim.GetNumAtoms() > 0:
                        # Call with no context for dummy to avoid recursive context passing
                        dummy_x = _calculate_atom_features_tensor(dummy_mol_for_dim, selected_atom_features)
                        atom_feature_dim = dummy_x.shape[1] if dummy_x.numel() > 0 else 0
                    else:
                        atom_feature_dim = 0
                        logger.warning(f"{log_prefix} Could not determine atom feature dimension from dummy molecule. Setting dimension to 0.")
                    pyg_data.x = torch.empty(0, atom_feature_dim, dtype=torch.float)
                except Exception as e:
                    raise StructuralFeatureError(
                        message="Failed to determine atom feature dimension for empty molecule.",
                        molecule_index=molecule_index,
                        inchi=inchi, 
                        feature_type="atom",
                        reason="Could not get feature dimension from dummy atom for empty molecule.",
                        detail=str(e)
                    ) from e
            else:
                pyg_data.x = _calculate_atom_features_tensor(rdkit_mol, selected_atom_features,
                                                               molecule_index=molecule_index, inchi=inchi) 
        else:
            logger.info(f"{log_prefix} No atom features configured to be added. Setting pyg_data.x to None.")
            pyg_data.x = None

        # Calculate and assign bond features (pyg_data.edge_attr)
        if selected_bond_features and hasattr(pyg_data, 'edge_index') and pyg_data.edge_index.size(1) > 0:
            pyg_data.edge_attr = _calculate_bond_features_tensor(rdkit_mol, pyg_data.edge_index, selected_bond_features,
                                                                 molecule_index=molecule_index, inchi=inchi) 
        else:
            logger.info(f"{log_prefix} No bond features configured, no edge_index present, or no edges. Setting pyg_data.edge_attr to None.")
            pyg_data.edge_attr = None

    except StructuralFeatureError:
        # Re-raise custom StructuralFeatureError directly as it already contains context
        raise
    except Exception as e:
        # Catch any other unexpected exceptions and wrap them in a StructuralFeatureError
        # to provide consistent error reporting for this module.
        raise StructuralFeatureError(
            message="An unexpected error occurred while adding structural features.",
            molecule_index=molecule_index,
            inchi=inchi, 
            reason="Unhandled exception during feature computation.",
            detail=str(e)
        ) from e

    return pyg_data
