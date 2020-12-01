from datetime import datetime
import logging
import sys

from pytz import timezone, utc


class LogHelper:
    """Helper class to set up logger."""
    log_format = '%(asctime)s %(levelname)s - %(name)s - %(message)s'

    @staticmethod
    def setup(log_path, level='INFO'):
        """Set up logger.

        Args:
            log_path (str): Path to create the log file.
            level (str): Logging level. Default: 'INFO'.
        """
        def custom_time(*args):
            utc_dt = utc.localize(datetime.utcnow())
            my_tz = timezone('EST')
            converted = utc_dt.astimezone(my_tz)
            return converted.timetuple()

        logging.basicConfig(
             filename=log_path,
             level=logging.getLevelName(level),
             format= LogHelper.log_format,
         )

        logging.Formatter.converter = custom_time

        # Set up logging to console
        console = logging.StreamHandler()
        console.setLevel(logging.DEBUG)
        console.setFormatter(logging.Formatter(LogHelper.log_format))
        # Add the console handler to the root logger
        logging.getLogger('').addHandler(console)

        # Log for unhandled exception
        logger = logging.getLogger(__name__)
        sys.excepthook = lambda *ex: logger.critical("Unhandled exception.", exc_info=ex)
