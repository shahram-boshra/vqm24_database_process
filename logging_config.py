# logging_config.py

"""
This module configures the logging system for the application.

It sets up console and file handlers, ensuring logs are written to both
standard output and a dedicated log file within the script's directory.
It also manages third-party library log levels to prevent excessive output.
"""
import logging
import sys
from pathlib import Path
import inspect # Needed for inspect.currentframe()
from typing import TextIO # For sys.stdout in StreamHandler

from exceptions import LoggingConfigurationError, BaseProjectError


def setup_logging() -> logging.Logger:
    """
    Configures and initializes the application's logging system.

    Sets up a root logger with both console (stdout) and file handlers.
    Logs are written to a file named after the main script, located in the
    same directory as the script. Prevents handler duplication on
    multiple calls. Also silences RDKit's logger to 'ERROR' level.

    Raises:
        LoggingConfigurationError: If there's an issue setting up
                                   file logging (e.g., permissions) or
                                   any other unexpected error during setup.

    Returns:
        logging.Logger: The configured application logger instance.
    """
    logger: logging.Logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO) # Set default logging level

    # Ensure handlers are not duplicated if called multiple times
    if not logger.handlers:
        try:
            # Get the name of the script without the .py extension
            # This part is generally robust, but path operations can sometimes fail
            script_path: Path = Path(inspect.getfile(inspect.currentframe()))
            log_file_name: str = script_path.with_suffix('.log').name
            log_file_path: Path = script_path.parent / log_file_name # Log file in the same directory

            # Create a formatter for both console and file output
            formatter: logging.Formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

            # Console Handler
            console_handler: logging.StreamHandler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

            # File Handler
            # This is where a FileHandler might encounter permissions issues or invalid path errors
            try:
                file_handler: logging.FileHandler = logging.FileHandler(log_file_path)
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
                logger.info(f"Logging initialized. Output will be written to console and '{log_file_path}'")
            except OSError as e:
                # Catch OS-related errors (e.g., permissions, invalid path) during file handler setup
                # Re-raise as our custom exception for a more consistent error handling strategy
                raise LoggingConfigurationError(
                    message=f"Failed to set up file logging to '{log_file_path}'.",
                    details=f"OS Error: {e}"
                ) from e # 'from e' links the original exception for better debugging

        except Exception as e:
            # Catch any other unexpected errors during the main setup process
            # Log the error and then re-raise it as our custom exception
            logger.error(f"An unexpected error occurred during logging setup: {e}", exc_info=True)
            raise LoggingConfigurationError(
                message="An unexpected error prevented logging from being fully initialized.",
                details=str(e)
            ) from e

    # Silence RDKit warnings by setting its logger level to ERROR
    rdkit_logger: logging.Logger = logging.getLogger('rdkit')
    rdkit_logger.setLevel(logging.ERROR)

    return logger
