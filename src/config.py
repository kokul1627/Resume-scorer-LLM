from hydra import initialize, compose
from typing import Any
import os
import sys

isFile = os.path.isfile("config.yaml")
if not isFile:
    sys.exit("***File missing:config file is not available in the path***")

def get_config_object(config_path: str = ".", config_name: str = "config.yaml") -> Any:
    """
    Initializes the Hydra configuration and composes the configuration object.

    Args:
        config_path (str): The path to the configuration directory. Defaults to "..".
        config_name (str): The name of the configuration file. Defaults to "config.yaml".

    Returns:
        Any: The composed configuration object.

    Raises:
        hydra.errors.MissingConfigException: If the configuration file or directory is not found.
    """
    with initialize(version_base=None, config_path=config_path):
        cfg = compose(config_name=config_name)
        return cfg