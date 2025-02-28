"""
Utilities for loading and processing Sparse Autoencoders (SAEs).
"""

import torch
import torch.nn as nn
import numpy as np
import os
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import json
import requests
from tqdm import tqdm
import importlib.util
import sys

logger = logging.getLogger(__name__)

def download_file(url: str, local_path: str) -> str:
    """
    Download a file from a URL to a local path with a progress bar.
    
    Args:
        url: URL to download
        local_path: Local path to save the file
        
    Returns:
        Path to the downloaded file
    """
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    # If file already exists, return its path
    if os.path.exists(local_path):
        logger.info(f"File already exists at {local_path}, skipping download")
        return local_path
    
    logger.info(f"Downloading {url} to {local_path}")
    
    # Stream the file content
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    # Get file size for progress bar
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 KB
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
    
    # Download with progress bar
    with open(local_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=block_size):
            if chunk:
                progress_bar.update(len(chunk))
                f.write(chunk)
    
    progress_bar.close()
    
    if total_size != 0 and progress_bar.n != total_size:
        logger.warning("Downloaded file size doesn't match expected size")
    
    return local_path

class SAELoader:
    """
    Utility class for loading and processing Sparse Autoencoders (SAEs).
    """
    
    def __init__(
        self,
        cache_dir: Optional[str] = "outputs/sae_cache",
    ):
        """
        Initialize the SAE loader.
        
        Args:
            cache_dir: Directory to cache downloaded SAEs
        """
        self.cache_dir = cache_dir
        
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
    
    def load_gemma_sae_decoder_weights(
        self,
        layer_indices: Optional[List[int]] = None,
        model_size: str = "2b",
        source: str = "gemma_scope",
    ) -> Dict[int, torch.Tensor]:
        """
        Load Gemma 2 SAE decoder weights for specified layers.
        
        Args:
            layer_indices: List of layer indices to load (if None, load all available)
            model_size: Model size ('2b' or '7b')
            source: Source of SAEs ('gemma_scope' or a path to local files)
            
        Returns:
            Dictionary mapping layer indices to decoder weights
        """
        # Determine layers to load
        if layer_indices is None:
            # By default, for Gemma 2 2B we'll focus on middle layers that are often
            # relevant for syntactic tasks like subject-verb agreement
            if model_size == "2b":
                # Layers 6-12 are often relevant for syntactic tasks
                layer_indices = list(range(6, 13))
            else:
                # For 7B, use proportionally similar layers
                layer_indices = list(range(10, 22))
        
        # Load decoder weights for each layer
        decoder_weights = {}
        
        if source == "gemma_scope":
            # Use SAELens to load pretrained SAEs
            
            from sae_lens import SAE
            
            for layer_idx in tqdm(layer_indices, desc="Loading SAE decoder weights"):
                try:
                    # Construct the correct SAE ID based on the layer index for canonical SAEs
                    sae_id = f"layer_{layer_idx}/width_16k/canonical"
                    logger.info(f"Loading SAE from gemma-scope-2b-pt-res-canonical, sae_id: {sae_id}")
                    
                    # Load the SAE for this layer using canonical SAEs
                    sae, cfg_dict, sparsity = SAE.from_pretrained(
                        release="gemma-scope-2b-pt-res-canonical",
                        sae_id=sae_id
                    )
                    
                    # Extract decoder weights
                    decoder = sae.W_dec
                    
                    # Store decoder weights
                    decoder_weights[layer_idx] = decoder
                    
                    logger.info(f"Loaded decoder weights for layer {layer_idx} with shape {decoder.shape}, avg L0: {sparsity}")
                
                except Exception as e:
                    logger.error(f"Error loading decoder weights for layer {layer_idx}: {e}")
        else:
            # For local files, use the path directly
            base_url = source
            layer_format = "layer_{layer_idx}.pt"
            
            for layer_idx in tqdm(layer_indices, desc="Loading SAE decoder weights"):
                try:
                    # Construct file path
                    file_name = layer_format.format(layer_idx=layer_idx)
                    local_path = os.path.join(base_url, file_name)
                    
                    # Load weights
                    weights = torch.load(local_path, map_location="cpu")
                    
                    # Extract decoder weights
                    if isinstance(weights, dict) and "decoder" in weights:
                        decoder = weights["decoder"]
                    elif isinstance(weights, dict) and "W_dec" in weights:
                        decoder = weights["W_dec"]
                    elif isinstance(weights, torch.Tensor):
                        # Assume the file contains only the decoder weights
                        decoder = weights
                    else:
                        raise ValueError(f"Unexpected format for SAE weights: {type(weights)}")
                    
                    # Store decoder weights
                    decoder_weights[layer_idx] = decoder
                    
                    logger.info(f"Loaded decoder weights for layer {layer_idx} with shape {decoder.shape}")
                
                except Exception as e:
                    logger.error(f"Error loading decoder weights for layer {layer_idx}: {e}")
        
        if not decoder_weights:
            raise ValueError(f"No decoder weights could be loaded for the specified layers: {layer_indices}")
        
        return decoder_weights
    
    def load_dummy_decoder_weights(
        self,
        layer_indices: List[int],
        hidden_size: int = 2048,
        n_features: int = 8192,
    ) -> Dict[int, torch.Tensor]:
        """
        Create dummy decoder weights for testing purposes.
        
        Args:
            layer_indices: List of layer indices to create weights for
            hidden_size: Size of hidden dimension in model
            n_features: Number of features in SAE
            
        Returns:
            Dictionary mapping layer indices to decoder weights
        """
        logger.warning("Using dummy decoder weights for testing purposes")
        
        decoder_weights = {}
        for layer_idx in layer_indices:
            # Initialize random weights
            decoder = torch.randn(hidden_size, n_features)
            decoder_weights[layer_idx] = decoder
        
        return decoder_weights
    
    def convert_sae_format(
        self,
        source_weights: Dict[str, Any],
        output_path: str,
        format_type: str = "gemma_scope"
    ) -> None:
        """
        Convert SAE weights from one format to another.
        
        Args:
            source_weights: Source weights to convert
            output_path: Path to save converted weights
            format_type: Target format type
        """
        if format_type == "gemma_scope":
            # Extract decoder weights
            if "W_dec" in source_weights:
                decoder = source_weights["W_dec"]
            elif "decoder" in source_weights:
                decoder = source_weights["decoder"]
            else:
                raise ValueError("Source weights do not contain decoder matrix under expected keys")
            
            # Create output directory
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save in target format
            torch.save({"decoder": decoder}, output_path)
            
            logger.info(f"Converted SAE weights to {format_type} format and saved to {output_path}") 