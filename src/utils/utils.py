"""
Utility functions for the analytical gradient approach.
"""

import torch
import numpy as np
import os
import json
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import pickle
from pathlib import Path

def setup_logging(log_file: Optional[str] = None, log_level:str = "INFO") -> None:
    """
    Set up logging configuration.
    
    Args:
        log_file: Path to log file (optional)
        level: Logging level
    """
    # Convert string log level to numeric level
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR
    }
    level = level_map.get(log_level.upper(), logging.INFO)

    handlers = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

def save_results(data: Any, file_path: str) -> None:
    """
    Save results to a file, determining the format based on file extension.
    
    Args:
        data: Data to save
        file_path: Path to save the data to
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    if file_path.endswith('.json'):
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    elif file_path.endswith('.pkl'):
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
    elif file_path.endswith('.pt') or file_path.endswith('.pth'):
        torch.save(data, file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

def load_results(file_path: str) -> Any:
    """
    Load results from a file, determining the format based on file extension.
    
    Args:
        file_path: Path to the data file
        
    Returns:
        The loaded data
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if file_path.endswith('.json'):
        with open(file_path, 'r') as f:
            return json.load(f)
    elif file_path.endswith('.pkl'):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    elif file_path.endswith('.pt') or file_path.endswith('.pth'):
        return torch.load(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

def rank_features(gradient_magnitudes: torch.Tensor, top_k: int = 20) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Rank features based on their gradient magnitudes.
    
    Args:
        gradient_magnitudes: Tensor of gradient magnitudes for features
        top_k: Number of top features to return
        
    Returns:
        Tuple of (indices, values) for the top_k features
    """
    if len(gradient_magnitudes.shape) > 1:
        # If we have a batch dimension, aggregate across it
        agg_magnitudes = gradient_magnitudes.abs().mean(dim=0)
    else:
        agg_magnitudes = gradient_magnitudes.abs()
    
    # Get top k features
    values, indices = torch.topk(agg_magnitudes, min(top_k, agg_magnitudes.shape[0]))
    return indices, values

def print_summary_statistics(data: torch.Tensor, name: str = "Data") -> None:
    """
    Print summary statistics for a tensor.
    
    Args:
        data: Tensor to summarize
        name: Name to include in the summary
    """
    logging.info(f"{name} - Shape: {data.shape}")
    logging.info(f"{name} - Mean: {data.mean().item():.6f}")
    logging.info(f"{name} - Std: {data.std().item():.6f}")
    logging.info(f"{name} - Min: {data.min().item():.6f}")
    logging.info(f"{name} - Max: {data.max().item():.6f}")
    logging.info(f"{name} - Nan count: {torch.isnan(data).sum().item()}")
    logging.info(f"{name} - Inf count: {torch.isinf(data).sum().item()}") 