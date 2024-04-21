import logging
import sys

from typing import Optional
from typing import List


class CustomFormatter(logging.Formatter):
    COLOR_RED      = "\033[91m"
    COLOR_GREEN    = "\033[92m"
    COLOR_YELLOW   = "\033[93m"
    COLOR_PURPLE   = "\033[94m"
    COLOR_RESET    = "\033[0m"
    COLOR_GREY     = "\033[90m"
    COLOR_DEFAULT  = "\033[38;21m"
    COLOR_BOLD_RED = "\033[1;31m"
    COLOR_BLUE     = "\033[38;5;39m"

    FORMATS = {
        logging.INFO:     COLOR_BLUE,
        logging.DEBUG:    COLOR_GREY,
        logging.WARNING:  COLOR_YELLOW,
        logging.ERROR:    COLOR_RED,
        logging.CRITICAL: COLOR_BOLD_RED
    }

    def formatTime(self, record, datefmt = None):
        asctime = super().formatTime(record, datefmt)
        return f"{self.COLOR_PURPLE}{asctime}{self.COLOR_RESET}"

    def format(self, record):
        color = self.FORMATS.get(record.levelno)
        record.levelname = color + record.levelname + self.COLOR_RESET
        return super().format(record)


def setup_logger(level: int = logging.INFO, filename: Optional[str] = None, mode: str = "w", **kwargs) -> None:
    """
    """
    basic_formatter = logging.Formatter(
        fmt = "%(asctime)s [%(levelname)s]: %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S"
    )
    color_formatter = CustomFormatter(
        fmt = "%(asctime)s [%(levelname)s]: %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S"
    )

    handlers: List[logging.Handler] = []

    # Create a console handler for command line output
    console_handler = logging.StreamHandler(stream = sys.stdout)
    console_handler.setFormatter(color_formatter)
    handlers.append(console_handler)

    if filename is not None:
        file_handler = logging.FileHandler(filename, mode = mode, **kwargs) 
        file_handler.setFormatter(basic_formatter)
        handlers.append(file_handler)

    logging.basicConfig(
        level = level,
        handlers = handlers
    )



