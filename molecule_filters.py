# molecule_filters.py

"""
This module provides functions for pre-filtering molecular data represented as PyG Data objects.

It includes filters based on atom count (min/max) and the presence/absence of specific heavy atoms.
Custom exceptions are used to differentiate between data processing errors and
intentional filter rejections.
"""

import logging
import torch
from torch_geometric.data import Data
from typing import Union, Dict, Set, List, Optional


from config import HEAVY_ATOM_SYMBOLS_TO_Z

from exceptions import (
    MoleculeProcessingError,
    MoleculeFilterRejectedError,
    AtomFilterError,
    ConfigurationError
)

logger = logging.getLogger(__name__)


def apply_pre_filters(
    pyg_data: Data,
    filter_config: Dict[str, Union[int, Dict[str, Union[str, List[str]]]]],
    logger: logging.Logger
) -> bool:
    """
    Applies a series of pre-filters to a PyG Data object.

    Molecules failing any filter will raise an exception:
    - **MoleculeProcessingError**: For fundamental data integrity issues (e.g., missing 'z').
    - **MoleculeFilterRejectedError**: For intentional rejections based on filter criteria.
    - **AtomFilterError**: For invalid heavy atom filter configurations.

    Args:
        pyg_data (torch_geometric.data.Data): The PyG Data object representing the molecule.
        filter_config (dict): A dictionary containing filter parameters (e.g., 'max_atoms', 'min_atoms',
                              'heavy_atom_filter').
        logger (logging.Logger): The logger instance for recording messages.

    Raises:
        MoleculeProcessingError: If critical molecular data (e.g., atomic numbers) is missing.
        MoleculeFilterRejectedError: If the molecule fails an explicit filter condition.
        AtomFilterError: If the heavy atom filter configuration is invalid.

    Returns:
        bool: **True** if the molecule successfully passes all active filters.
              (Note: This return is only reached if no exceptions are raised.)
    """
    mol_idx: Union[int, str] = getattr(pyg_data, 'original_mol_idx', 'N/A')
    smiles: str = getattr(pyg_data, 'smiles', 'N/A')
    num_nodes: Union[int, str] = getattr(pyg_data, 'num_nodes', 'N/A')

    # Max Atoms Filter
    max_atoms: Optional[int] = filter_config.get('max_atoms')
    if max_atoms is not None and num_nodes != 'N/A' and num_nodes > max_atoms:
        # Raise MoleculeFilterRejectedError for intentional filtering
        raise MoleculeFilterRejectedError(
            molecule_index=mol_idx,
            smiles=smiles,
            reason=f"Molecule excluded due to 'max_atoms' filter: {num_nodes} atoms exceeds max_atoms={max_atoms}."
        )

    # Min Atoms Filter
    min_atoms: Optional[int] = filter_config.get('min_atoms')
    if min_atoms is not None and num_nodes != 'N/A' and num_nodes < min_atoms:
        # Raise MoleculeFilterRejectedError for intentional filtering
        raise MoleculeFilterRejectedError(
            molecule_index=mol_idx,
            smiles=smiles,
            reason=f"Molecule excluded due to 'min_atoms' filter: {num_nodes} atoms is below min_atoms={min_atoms}."
        )

    # --- Heavy Atom Filtering ---
    heavy_atom_filter: Optional[Dict[str, Union[str, List[str]]]] = filter_config.get('heavy_atom_filter')
    if heavy_atom_filter:
        mode: Optional[str] = heavy_atom_filter.get('mode') # 'include' or 'exclude'
        filter_symbols: List[str] = heavy_atom_filter.get('atoms', [])

        if not mode:
            raise AtomFilterError(
                message="Heavy atom filter configured but 'mode' is missing.",
                config_key="filter_config.heavy_atom_filter.mode",
                detail=f"Molecule {mol_idx} (SMILES: {smiles}) was processed, but filter could not be applied due to configuration error."
            )
        if not filter_symbols:
            raise AtomFilterError(
                message="Heavy atom filter configured but 'atoms' list is missing or empty.",
                config_key="filter_config.heavy_atom_filter.atoms",
                detail=f"Molecule {mol_idx} (SMILES: {smiles}) was processed, but filter could not be applied due to configuration error."
            )
        else:
            target_heavy_zs_for_filter: Set[int] = set()
            for symbol in filter_symbols:
                normalized_symbol: Optional[str] = None
                if len(symbol) == 1:
                    normalized_symbol = symbol.upper()
                elif len(symbol) == 2:
                    normalized_symbol = symbol[0].upper() + symbol[1].lower()
                else:
                    raise AtomFilterError(
                        message=f"Invalid atom symbol '{symbol}' in heavy atom filter configuration: unexpected symbol length.",
                        config_key="filter_config.heavy_atom_filter.atoms",
                        invalid_atom_symbol=symbol
                    )

                atomic_num: Optional[int] = HEAVY_ATOM_SYMBOLS_TO_Z.get(normalized_symbol)
                if atomic_num:
                    target_heavy_zs_for_filter.add(atomic_num)
                else:
                    raise AtomFilterError(
                        message=f"Unknown heavy atom symbol '{symbol}' in filter configuration.",
                        config_key="filter_config.heavy_atom_filter.atoms",
                        invalid_atom_symbol=symbol
                    )

            if not target_heavy_zs_for_filter:
                raise AtomFilterError(
                    message="Heavy atom filter configured, but no valid atomic symbols were provided or recognized.",
                    config_key="filter_config.heavy_atom_filter.atoms",
                    detail="Please check the 'atoms' list in your filter configuration."
                )
            else:
                if not hasattr(pyg_data, 'z') or pyg_data.z is None or pyg_data.z.numel() == 0:
                     # This is a genuine data integrity issue, not a user-defined filter rejection
                     raise MoleculeProcessingError(
                         molecule_index=mol_idx,
                         smiles=smiles,
                         reason=f"Cannot apply heavy atom filter: 'z' (atomic numbers) is missing or empty in PyG Data."
                     )

                molecule_all_zs: List[int] = pyg_data.z.unique().tolist()
                molecule_heavy_zs: Set[int] = {z for z in molecule_all_zs if z != 1} # Exclude Hydrogen

                if not molecule_heavy_zs:
                    if mode == 'include':
                        # Raise MoleculeFilterRejectedError for intentional filtering
                        raise MoleculeFilterRejectedError(
                            molecule_index=mol_idx,
                            smiles=smiles,
                            reason=f"Molecule excluded due to 'heavy_atom_filter' ('include' mode): Molecule has no heavy atoms, but filter requires {filter_symbols}."
                        )
                    elif mode == 'exclude':
                        pass # Passes, no heavy atoms to exclude
                else:
                    if mode == 'include':
                        if not molecule_heavy_zs.issubset(target_heavy_zs_for_filter):
                            unallowed_atoms: List[int] = list(molecule_heavy_zs - target_heavy_zs_for_filter)
                            # Raise MoleculeFilterRejectedError for intentional filtering
                            raise MoleculeFilterRejectedError(
                                molecule_index=mol_idx,
                                smiles=smiles,
                                reason=f"Molecule excluded due to 'heavy_atom_filter' ('include' mode): Molecule contains unallowed heavy atoms {unallowed_atoms}. Filtered for {filter_symbols}."
                            )
                    elif mode == 'exclude':
                        overlap_atoms: List[int] = list(molecule_heavy_zs.intersection(target_heavy_zs_for_filter))
                        if len(overlap_atoms) > 0:
                            # Raise MoleculeFilterRejectedError for intentional filtering
                            raise MoleculeFilterRejectedError(
                                molecule_index=mol_idx,
                                smiles=smiles,
                                reason=f"Molecule excluded due to 'heavy_atom_filter' ('exclude' mode) because it contains excluded heavy atom(s) {overlap_atoms}. Filtered against {filter_symbols}."
                            )
                    else:
                        raise AtomFilterError(
                            message=f"Unknown heavy atom filter mode '{mode}'. Expected 'include' or 'exclude'.",
                            config_key="filter_config.heavy_atom_filter.mode"
                        )

    return True # The molecule passes all applied filters
