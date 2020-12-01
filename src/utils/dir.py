from datetime import datetime
import logging
import os
import pathlib

from pytz import timezone


_logger = logging.getLogger(__name__)


def create_dir(output_dir):
    """Create directory.

    Args:
        output_dir (str): A directory to create if not found.

    Returns:
        exit_code: 0 (success) or -1 (failed).
    """
    try:
        if not os.path.exists(output_dir):
            pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

        return 0
    except Exception as err:
        _logger.critical("Error when creating directory: {}.".format(err))
        exit(-1)


def get_datetime_str():
    """Get string based on current datetime."""
    return datetime.now(timezone('EST')).strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3]
