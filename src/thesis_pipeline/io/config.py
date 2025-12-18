"""
Configuration loading utilities for thesis pipeline.
"""
from pathlib import Path
from typing import Any, Dict
import yaml


def load_config(config_name: str, configs_dir: Path = None) -> Dict[str, Any]:
    """
    Load a YAML configuration file.
    
    Args:
        config_name: Name of config file (without .yaml extension)
        configs_dir: Path to configs directory (defaults to repo root/configs)
    
    Returns:
        Dictionary containing configuration
    """
    if configs_dir is None:
        configs_dir = Path(__file__).parent.parent.parent / "configs"
    
    config_path = configs_dir / f"{config_name}.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def load_all_configs(configs_dir: Path = None) -> Dict[str, Dict[str, Any]]:
    """
    Load all configuration files.
    
    Returns:
        Dictionary with keys: 'global', 'reddit', 'news'
    """
    return {
        'global': load_config('global', configs_dir),
        'reddit': load_config('reddit_politosphere', configs_dir),
        'news': load_config('news_ccnews', configs_dir)
    }
