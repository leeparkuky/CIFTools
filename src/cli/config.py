# %%
# this script reads config yaml for CIFTools
# and sets up the configuration for the CLI
import os
import sys
import yaml
from typing import Dict, Any, Optional
from pathlib import Path


# reading yaml config file
def read_yaml_config(config_path: str) -> Dict[str, Any]:
    """
    Reads a YAML configuration file and returns the contents as a dictionary.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        Dict[str, Any]: Contents of the YAML file as a dictionary.
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


# example


# %%
if __name__ == "__main__":
    # get the path to the config file
    config_path = "../../example_config/kentucky.yaml"
    # read the config file
    config = read_yaml_config(config_path)
    # print the config
    print(config)
