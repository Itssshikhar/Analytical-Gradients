"""
Analytical Gradient implementation for Sparse Feature Circuits.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import numpy as np
import os
import logging
from tqdm import tqdm
import time
from pathlib import Path
import json

from src.utils.utils import rank_features, print_summary_statistics, save_results

logger = logging.getLogger(__name__)

class AnalyticalGradient:
    """
    Implementation of Analytical Gradient approach for Sparse Feature Circuits.
    
    This class computes gradients on residual streams directly and multiplies by
    the decoder matrix to approximate SAE activation gradients.
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        sae_decoder_weights: Dict[int, torch.Tensor],
        layer_indices: Optional[List[int]] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.float16,
        cache_dir: Optional[str] = None,
        save_interval: int = 10,
    ):
        """
        Initialize the Analytical Gradient approach.
        
        Args:
            model: Pretrained language model
            tokenizer: Tokenizer for the model
            sae_decoder_weights: Dictionary mapping layer indices to SAE decoder weights
            layer_indices: List of layer indices to compute gradients for (if None, use all layers in sae_decoder_weights)
            device: Device to run computations on
            dtype: Data type for computations
            cache_dir: Directory to cache intermediate results
            save_interval: How often to save intermediate results (in number of examples)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.sae_decoder_weights = {
            layer_idx: weights.to(device=device, dtype=dtype)
            for layer_idx, weights in sae_decoder_weights.items()
        }
        self.layer_indices = layer_indices or sorted(list(sae_decoder_weights.keys()))
        self.device = device
        self.dtype = dtype
        self.cache_dir = cache_dir
        self.save_interval = save_interval
        
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        
        # Register hooks for capturing residual stream values and gradients
        self.handles = []
        self.residual_streams = {}
        self.residual_grads = {}
        self._register_hooks()
        
        logger.info(f"Analytical Gradient initialized with {len(self.layer_indices)} layers")
        logger.info(f"Using device: {device}, dtype: {dtype}")
    
    def _register_hooks(self):
        """
        Register hooks to capture residual stream values and gradients.
        """
        # Clear any existing hooks
        self._remove_hooks()
        
        # Reset storage
        self.residual_streams = {layer_idx: None for layer_idx in self.layer_indices}
        self.residual_grads = {layer_idx: None for layer_idx in self.layer_indices}
        
        # Define forward hook
        def forward_hook(layer_idx):
            def hook(module, input, output):
                self.residual_streams[layer_idx] = output.detach().clone().requires_grad_(True)
                return self.residual_streams[layer_idx]
            return hook
        
        # Register hooks for each target layer
        for layer_idx in self.layer_indices:
            # Get the specific layer (implementation dependent on model architecture)
            # For Gemma 2, we need to access the appropriate transformer layer output
            # For example: self.model.model.layers[layer_idx].output
            # The exact path depends on the model's architecture
            
            # For Gemma 2, use this path (adjust based on actual model structure)
            residual_layer = self.model.model.layers[layer_idx].input_layernorm
            
            # Register forward hook
            handle = residual_layer.register_forward_hook(forward_hook(layer_idx))
            self.handles.append(handle)
    
    def _remove_hooks(self):
        """
        Remove all registered hooks.
        """
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def backward_hook(self, layer_idx):
        """
        Create a backward hook function for the specified layer.
        
        Args:
            layer_idx: Layer index to create hook for
            
        Returns:
            Hook function that stores gradients
        """
        def hook(grad):
            self.residual_grads[layer_idx] = grad.detach().clone()
            return grad
        return hook
    
    def compute_gradients(self, input_ids, target_tokens=None, loss_fn=None):
        """
        Compute analytical gradients for a single input.
        
        Args:
            input_ids: Input token IDs
            target_tokens: Target tokens for computing loss (if None, use next token prediction)
            loss_fn: Custom loss function (if None, use cross-entropy loss)
            
        Returns:
            Dictionary mapping layer indices to feature gradients
        """
        # Move input to device
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.to(self.device)
        else:
            input_ids = torch.tensor(input_ids, device=self.device)
        
        # Ensure input has batch dimension
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        
        # Forward pass with model
        outputs = self.model(input_ids)
        logits = outputs.logits

        # Debug info about residual streams
        logger.debug(f"Captured residual streams for layers: {list(self.residual_streams.keys())}")
        for layer_idx, stream in self.residual_streams.items():
            if stream is not None:
                logger.debug(f"Layer {layer_idx} residual stream shape: {stream.shape}, requires_grad: {stream.requires_grad}")
            else:
                logger.debug(f"Layer {layer_idx} residual stream is None")
        
        # Register backward hooks for each layer
        for layer_idx in self.layer_indices:
            if self.residual_streams[layer_idx] is not None:
                self.residual_streams[layer_idx].register_hook(
                    self.backward_hook(layer_idx)
                )
        
        # Compute loss
        if loss_fn is not None:
            loss = loss_fn(logits, input_ids)
        else:
            # Default: next token prediction loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            
            # If target_tokens is provided, focus only on those positions
            if target_tokens is not None:
                if isinstance(target_tokens, int):
                    target_tokens = [target_tokens]
                
                # Extract logits and labels at target positions
                target_logits = torch.stack([shift_logits[:, pos, :] for pos in target_tokens], dim=1)
                target_labels = torch.stack([shift_labels[:, pos] for pos in target_tokens], dim=1)
                
                loss = F.cross_entropy(
                    target_logits.view(-1, target_logits.size(-1)),
                    target_labels.view(-1)
                )
            else:
                # Compute loss on all tokens
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )

        logger.debug(f"Computed loss: {loss.item()}")
        
        # Backward pass
        loss.backward()

        # Debug info about gradients
        logger.debug(f"Captured gradients for layers: {list(self.residual_grads.keys())}")
        for layer_idx, grad in self.residual_grads.items():
            if grad is not None:
                logger.debug(f"Layer {layer_idx} gradient shape: {grad.shape}, non-zero elements: {torch.count_nonzero(grad).item()}")
            else:
                logger.debug(f"Layer {layer_idx} gradient is None")
        
        # Compute feature gradients for each layer
        feature_gradients = {}
        for layer_idx in self.layer_indices:
            if self.residual_grads[layer_idx] is not None and self.sae_decoder_weights.get(layer_idx) is not None:
                # Get residual stream gradient (∇R_l)
                grad_residual = self.residual_grads[layer_idx]
                
                # Get decoder matrix (W_dec)
                decoder = self.sae_decoder_weights[layer_idx]

                # Debug info about decoder
                logger.debug(f"Layer {layer_idx} decoder shape: {decoder.shape}")
                
                # Reshape for matrix multiplication
                batch_size, seq_len, hidden_dim = grad_residual.shape
                grad_residual_flat = grad_residual.reshape(-1, hidden_dim)
                
                # Debug info about decoder shape
                logger.debug(f"Layer {layer_idx} decoder shape: {decoder.shape}")
                
                # Transpose decoder to make dimensions compatible for matrix multiplication
                # If decoder is [n_features, hidden_dim], transpose to [hidden_dim, n_features]
                if decoder.shape[0] != hidden_dim:
                    decoder = decoder.T
                    logger.debug(f"Transposed decoder shape to: {decoder.shape}")
                
                # Compute gradient approximation: ∇Z_l ≈ ∇R_l * W_dec
                grad_features = torch.matmul(grad_residual_flat, decoder)
                
                # Reshape back to [batch_size, seq_len, n_features]
                grad_features = grad_features.reshape(batch_size, seq_len, -1)

                # Debug info about feature gradients
                logger.debug(f"Layer {layer_idx} feature gradients shape: {grad_features.shape}")
                logger.debug(f"Layer {layer_idx} feature gradients stats: min={grad_features.min().item():.6f}, max={grad_features.max().item():.6f}, mean={grad_features.mean().item():.6f}")
                
                feature_gradients[layer_idx] = grad_features
            else:
                logger.warning(f"Missing gradient or decoder for layer {layer_idx}")
        
        # Clear gradients
        self.model.zero_grad()
        
        return feature_gradients
    
    def analyze_dataset(
        self,
        dataset,
        target_token_key="verb_position",
        batch_size=1,
        top_k=20,
        save_prefix="analytical_gradient_results"
    ):
        """
        Analyze a dataset and compute feature importance.
        
        Args:
            dataset: Dataset to analyze
            target_token_key: Key in dataset for the target token position
            batch_size: Batch size for processing
            top_k: Number of top features to return per layer
            save_prefix: Prefix for saved result files
            
        Returns:
            Dictionary with feature importance and related information
        """
        start_time = time.time()
        logger.info(f"Starting analysis of {len(dataset)} examples")
        
        # Initialize storage for accumulated gradients
        acc_gradients = {}
        
        # Process examples
        for idx in tqdm(range(0, len(dataset), batch_size), desc="Processing examples"):
            batch_indices = slice(idx, min(idx + batch_size, len(dataset)))
            batch = dataset[batch_indices]
            
            # Get input_ids and target positions
            input_ids = batch["input_ids"]
            
            # Get target token positions if available
            if target_token_key in batch and batch[target_token_key] is not None:
                target_positions = batch[target_token_key]
            else:
                target_positions = None
            
            # Compute gradients
            feature_gradients = self.compute_gradients(
                input_ids=input_ids,
                target_tokens=target_positions
            )
            
            # Accumulate absolute gradients for each layer
            for layer_idx, gradients in feature_gradients.items():
                # Average over batch and sequence dimensions, then take absolute value
                avg_gradients = gradients.abs().mean(dim=(0, 1))
                
                # Initialize acc_gradients[layer_idx] if not already done
                if layer_idx not in acc_gradients:
                    acc_gradients[layer_idx] = torch.zeros_like(avg_gradients)
                    logger.debug(f"Initialized acc_gradients for layer {layer_idx} with shape: {acc_gradients[layer_idx].shape}")
                
                # Accumulate
                acc_gradients[layer_idx] += avg_gradients
                
            # Save intermediate results periodically
            if (idx // batch_size) % self.save_interval == 0 and idx > 0:
                logger.info(f"Processed {idx + batch_size} examples. Saving intermediate results...")
                self._save_intermediate_results(acc_gradients, idx + batch_size, save_prefix)
        
        # Process results
        results = self._process_results(acc_gradients, top_k)
        
        # Save final results
        output_path = os.path.join(
            self.cache_dir if self.cache_dir else "outputs",
            f"{save_prefix}_final.json"
        )
        save_results(results, output_path)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Analysis completed in {elapsed_time:.2f} seconds")
        logger.info(f"Results saved to {output_path}")
        
        return results
    
    def _save_intermediate_results(self, acc_gradients, num_examples, save_prefix):
        """
        Save intermediate results during dataset analysis.
        
        Args:
            acc_gradients: Accumulated gradients for each layer
            num_examples: Number of examples processed so far
            save_prefix: Prefix for saved file
        """
        intermediate_results = {}
        
        for layer_idx, gradients in acc_gradients.items():
            # Get top features for this layer
            indices, values = rank_features(gradients, top_k=20)
            
            # Convert to CPU for saving
            indices_cpu = indices.detach().cpu().numpy().tolist()
            values_cpu = values.detach().cpu().numpy().tolist()
            
            intermediate_results[f"layer_{layer_idx}"] = {
                "top_feature_indices": indices_cpu,
                "top_feature_values": values_cpu
            }
        
        # Add metadata
        intermediate_results["metadata"] = {
            "num_examples_processed": num_examples,
            "layers_analyzed": self.layer_indices,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save to file
        output_path = os.path.join(
            self.cache_dir if self.cache_dir else "outputs",
            f"{save_prefix}_intermediate_{num_examples}.json"
        )
        
        save_results(intermediate_results, output_path)
    
    def _process_results(self, acc_gradients, top_k):
        """
        Process accumulated gradients to get final results.
        
        Args:
            acc_gradients: Accumulated gradients for each layer
            top_k: Number of top features to return
            
        Returns:
            Dictionary with processed results
        """
        results = {}
        
        # Process each layer
        for layer_idx, gradients in acc_gradients.items():
            # Get top features
            indices, values = rank_features(gradients, top_k=top_k)
            
            # Convert to CPU for saving
            indices_cpu = indices.detach().cpu().numpy().tolist()
            values_cpu = values.detach().cpu().numpy().tolist()
            
            # Store in results
            results[f"layer_{layer_idx}"] = {
                "top_feature_indices": indices_cpu,
                "top_feature_values": values_cpu,
                "decoder_shape": self.sae_decoder_weights[layer_idx].shape
            }
        
        # Add metadata
        results["metadata"] = {
            "layers_analyzed": self.layer_indices,
            "top_k": top_k,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return results
    
    def __del__(self):
        """
        Clean up resources when the object is destroyed.
        """
        self._remove_hooks() 