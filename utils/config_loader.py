import yaml
import os

def load_config(config_path:str = None) -> dict:
    if config_path is None:
        config_path = os.path.join("config", "config.yaml")
    
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    print(config)
    return config

load_config("config\config.yaml")