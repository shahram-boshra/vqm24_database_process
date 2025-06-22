# exceptions.py

from typing import Optional, Type


class BaseProjectError(Exception):
    """Base exception for all custom errors in this project."""
    pass


class LoggingConfigurationError(BaseProjectError):
    """Exception raised for errors specifically during logging configuration."""
    def __init__(self, message: str = "Error configuring logging.", details: Optional[str] = None) -> None:
        """
        Initializes the LoggingConfigurationError.

        Args:
            message (str): A general description of the logging configuration error.
            details (Optional[str]): More specific details about the error, if available.
        """
        self.message: str = message
        self.details: Optional[str] = details
        super().__init__(self.message)

    def __str__(self) -> str:
        msg: str = f"{self.message}"
        if self.details:
            msg += f" Details: {self.details}"
        return msg


class ConfigurationError(BaseProjectError):
    """Exception raised for errors in configuration files (e.g., config.yaml)."""
    def __init__(self, message: str = "Invalid configuration.", config_key: Optional[str] = None, expected_type: Optional[Type] = None, actual_value: Optional[object] = None) -> None:
        """
        Initializes the ConfigurationError.

        Args:
            message (str): A general description of the configuration error.
            config_key (Optional[str]): The specific key in the configuration that caused the error.
            expected_type (Optional[Type]): The expected data type for the configuration key.
            actual_value (Optional[object]): The actual value found for the configuration key.
        """
        self.message: str = message
        self.config_key: Optional[str] = config_key
        self.expected_type: Optional[Type] = expected_type
        self.actual_value: Optional[object] = actual_value
        super().__init__(self.message)

    def __str__(self) -> str:
        msg: str = f"{self.message}"
        if self.config_key:
            msg += f" Key: '{self.config_key}'"
        if self.expected_type:
            msg += f", Expected Type: {self.expected_type.__name__}"
        if self.actual_value is not None:
            msg += f", Actual Value: '{self.actual_value}' (Type: {type(self.actual_value).__name__})"
        return msg


class DataProcessingError(BaseProjectError):
    """Exception raised for errors encountered during data processing."""
    def __init__(self, message: str = "Error during data processing.", item_identifier: Optional[str] = None, details: Optional[str] = None) -> None:
        """
        Initializes the DataProcessingError.

        Args:
            message (str): A general description of the data processing error.
            item_identifier (Optional[str]): An identifier for the specific data item that caused the error.
            details (Optional[str]): More specific details about the error.
        """
        self.message: str = message
        self.item_identifier: Optional[str] = item_identifier
        self.details: Optional[str] = details
        super().__init__(self.message)

    def __str__(self) -> str:
        msg: str = f"{self.message}"
        if self.item_identifier:
            msg += f" Item ID: '{self.item_identifier}'"
        if self.details:
            msg += f", Details: {self.details}"
        return msg


class MoleculeProcessingError(DataProcessingError): # Inherit from DataProcessingError
    """Custom exception for errors encountered during molecule processing."""
    def __init__(self, message: str = "Error processing molecule.", molecule_index: Optional[int] = None, smiles: Optional[str] = None, reason: Optional[str] = None, detail: Optional[str] = None) -> None:
        """
        Initializes the MoleculeProcessingError.

        Args:
            message (str): A general description of the molecule processing error.
            molecule_index (Optional[int]): The index of the molecule in the dataset, if applicable.
            smiles (Optional[str]): The SMILES string of the molecule that caused the error.
            reason (Optional[str]): A brief explanation of why the processing failed.
            detail (Optional[str]): Further technical or specific details about the failure.
        """
        # Construct a more specific message for the parent DataProcessingError
        full_message: str = message
        if molecule_index is not None and smiles is not None and reason is not None:
            full_message = f"Mol {molecule_index} (SMILES: {smiles}): {reason}"
            if detail:
                full_message += f" Detail: {detail}"
        elif reason: # If only reason is provided
            full_message = f"Error processing molecule: {reason}"


        super().__init__(message=full_message, item_identifier=f"Mol {molecule_index}" if molecule_index is not None else None, details=reason)
        self.molecule_index: Optional[int] = molecule_index
        self.smiles: Optional[str] = smiles
        self.reason: Optional[str] = reason
        self.detail: Optional[str] = detail

    def __str__(self) -> str:
        # Override to ensure the specific molecule-related info is always present
        msg: str = f"Error processing molecule"
        if self.molecule_index is not None:
            msg += f" (Index: {self.molecule_index}"
            if self.smiles:
                msg += f", SMILES: {self.smiles}"
            msg += ")"
        
        if self.reason:
            msg += f": {self.reason}"
        elif self.message: # Fallback to general message if no specific reason
             msg += f": {self.message}"

        if self.detail:
            msg += f". Details: {self.detail}"
        return msg


class MoleculeFilterRejectedError(MoleculeProcessingError):
    """
    Exception raised when a molecule is explicitly rejected by a pre-defined filter.

    This exception signifies an expected outcome based on filtering rules,
    rather than an unexpected processing failure.
    """
    def __init__(self, molecule_index: int, smiles: str, reason: str, detail: Optional[str] = None) -> None:
        """
        Initializes the MoleculeFilterRejectedError.

        Args:
            molecule_index (int): The index of the molecule in the dataset.
            smiles (str): The SMILES string of the rejected molecule.
            reason (str): The specific filter criterion that led to the rejection.
            detail (Optional[str]): Additional details about the rejection, if any.
        """
        # When raising this, the 'reason' should clearly state 'filter rejected'
        super().__init__(
            message=f"Molecule rejected by filter.", # A more general message for the parent
            molecule_index=molecule_index,
            smiles=smiles,
            reason=reason, # Specific reason for rejection
            detail=detail
        )
        # No additional attributes needed, as parent handles them.
        # The __str__ from MoleculeProcessingError will be sufficient and correctly
        # format the 'reason' as the specific rejection criterion.


class MissingDependencyError(BaseProjectError):
    """Exception raised when a required dependency (e.g., PyG transform) is missing."""
    def __init__(self, message: str = "A required dependency is missing or not importable.", dependency_name: Optional[str] = None) -> None:
        """
        Initializes the MissingDependencyError.

        Args:
            message (str): A general message indicating a missing dependency.
            dependency_name (Optional[str]): The name of the specific dependency that is missing.
        """
        self.message: str = message
        self.dependency_name: Optional[str] = dependency_name
        super().__init__(self.message)

    def __str__(self) -> str:
        msg: str = f"{self.message}"
        if self.dependency_name:
            msg += f" Dependency: '{self.dependency_name}'"
        return msg


class AtomFilterError(ConfigurationError):
    """Exception raised specifically for errors in the heavy_atom_filter configuration."""
    def __init__(self, message: str = "Invalid heavy atom filter configuration.", config_key: Optional[str] = None, invalid_atom_symbol: Optional[str] = None) -> None:
        """
        Initializes the AtomFilterError.

        Args:
            message (str): A general description of the atom filter configuration error.
            config_key (Optional[str]): The specific configuration key related to the atom filter.
            invalid_atom_symbol (Optional[str]): The atom symbol that was found to be invalid.
        """
        self.message: str = message
        self.config_key: Optional[str] = config_key
        self.invalid_atom_symbol: Optional[str] = invalid_atom_symbol
        super().__init__(self.message, config_key=config_key) # Pass config_key to parent

    def __str__(self) -> str:
        msg: str = f"{self.message}"
        if self.config_key:
            msg += f" Key: '{self.config_key}'"
        if self.invalid_atom_symbol:
            msg += f", Invalid Atom Symbol: '{self.invalid_atom_symbol}'"
        return msg


class RDKitConversionError(MoleculeProcessingError):
    """Exception raised when RDKit fails to create a molecule or conformer."""
    def __init__(self, message: str = "Failed to create RDKit molecule or conformer.", molecule_index: Optional[int] = None, smiles: Optional[str] = None, reason: Optional[str] = None, detail: Optional[str] = None) -> None:
        """
        Initializes the RDKitConversionError.

        Args:
            message (str): A general message about the RDKit conversion failure.
            molecule_index (Optional[int]): The index of the molecule.
            smiles (Optional[str]): The SMILES string of the molecule.
            reason (Optional[str]): The reason for the RDKit conversion failure.
            detail (Optional[str]): Specific details about the RDKit error.
        """
        super().__init__(message=message, molecule_index=molecule_index, smiles=smiles, reason=reason, detail=detail)


class PyGDataCreationError(MoleculeProcessingError):
    """Exception raised when there's an issue creating the PyTorch Geometric Data object."""
    def __init__(self, message: str = "Failed to create PyTorch Geometric Data object.", molecule_index: Optional[int] = None, smiles: Optional[str] = None, reason: Optional[str] = None, detail: Optional[str] = None) -> None:
        """
        Initializes the PyGDataCreationError.

        Args:
            message (str): A general message about the PyTorch Geometric Data object creation failure.
            molecule_index (Optional[int]): The index of the molecule.
            smiles (Optional[str]): The SMILES string of the molecule.
            reason (Optional[str]): The reason for the PyG Data creation failure.
            detail (Optional[str]): Specific details about the PyG Data error.
        """
        super().__init__(message=message, molecule_index=molecule_index, smiles=smiles, reason=reason, detail=detail)


class PropertyEnrichmentError(MoleculeProcessingError):
    """Exception raised when an error occurs during property enrichment."""
    def __init__(self, message: str = "Failed to enrich PyG Data with properties.", molecule_index: Optional[int] = None, smiles: Optional[str] = None, property_name: Optional[str] = None, reason: Optional[str] = None, detail: Optional[str] = None) -> None:
        """
        Initializes the PropertyEnrichmentError.

        Args:
            message (str): A general message about the property enrichment failure.
            molecule_index (Optional[int]): The index of the molecule.
            smiles (Optional[str]): The SMILES string of the molecule.
            property_name (Optional[str]): The name of the property that failed to be enriched.
            reason (Optional[str]): The reason for the property enrichment failure.
            detail (Optional[str]): Specific details about the enrichment error.
        """
        super().__init__(message=message, molecule_index=molecule_index, smiles=smiles, reason=reason, detail=detail)
        self.property_name: Optional[str] = property_name
    
    def __str__(self) -> str:
        msg: str = super().__str__()
        if self.property_name:
            msg += f" (Property: {self.property_name})"
        return msg
    
