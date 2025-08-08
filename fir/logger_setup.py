import logging
from pathlib import Path


def setup_logger(
    log_file: str = "output.txt", log_level: str = "INFO", to_console: bool = True
) -> logging.Logger:
    logger = logging.getLogger("fir_logger")
    logger.setLevel(log_level)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Only add handlers if they haven't been added yet
    if not logger.handlers:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        if to_console:
            ch = logging.StreamHandler()
            ch.setFormatter(formatter)
            logger.addHandler(ch)

    return logger
