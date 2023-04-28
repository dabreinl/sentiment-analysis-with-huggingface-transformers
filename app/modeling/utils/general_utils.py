import json
import logging


def load_config(config_file: str) -> json:
    with open(config_file, "r") as f:
        config = json.load(f)
    return config


def setup_logger(
    name: str = "training_pipeline",
    level: int = logging.DEBUG,
    log_file: str = "artifacts/output.log",
):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create a console handler with a custom formatter
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)

    # Create a file handler with the same custom formatter
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
