"""
Main script for running the analytical gradient approach for SFC and generating visualizations.
"""

import os
import sys
import argparse
import logging
import json
import torch
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Optional

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import utility functions and classes
from utils.utils import setup_logging, save_results, load_results
from data.dataset import SVADataset
from model.analytical_gradient import AnalyticalGradient
from model.sae_utils import SAELoader
from utils.visualization import (
    plot_model_performance,
    plot_layer_importance,
    plot_top_features,
    plot_feature_importance_comparison,
    plot_importance_distribution,
    plot_dataset_distribution
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Analytical Gradient for Sparse Feature Circuits and Visualization")
    
    # General arguments
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save outputs")
    parser.add_argument("--cache_dir", type=str, default="outputs/cache", help="Directory to cache intermediate results")
    parser.add_argument("--log_file", type=str, default="outputs/analytical_gradient.log", help="Path to log file")
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Model and SAE arguments
    parser.add_argument("--model_name", type=str, default="google/gemma-2-2b", help="Hugging Face model name")
    parser.add_argument("--sae_source", type=str, default="gemma_scope", help="Source of SAE weights ('gemma_scope', 'dummy', or path to local files)")
    parser.add_argument("--layers", type=str, default="6,7,8,9,10,11,12", help="Comma-separated list of layers to analyze")
    
    # Dataset arguments
    parser.add_argument("--num_examples", type=int, default=200, help="Number of examples to generate")
    parser.add_argument("--test_size", type=float, default=0.2, help="Fraction of examples to use for testing")
    parser.add_argument("--correct_only", action="store_true", help="Only use grammatically correct examples")
    
    # Analysis arguments
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for processing")
    parser.add_argument("--top_k", type=int, default=20, help="Number of top features to return per layer")
    parser.add_argument("--save_interval", type=int, default=10, help="Number of batches between saving intermediate results")
    
    # Runtime arguments
    parser.add_argument("--dtype", type=str, default="float16", help="Data type for computations")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on")
    
    # Visualization arguments
    parser.add_argument("--results_file", type=str, default="combined_results.json", help="Name of the results file")
    parser.add_argument("--viz_output_dir", type=str, default=None, help="Directory to save visualizations (defaults to output_dir/viz)")
    
    return parser.parse_args()


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    
    # Setup logging
    setup_logging(args.log_file, log_level=args.log_level)
    logger = logging.getLogger(__name__)
    logger.info(f"Arguments: {args}")
    
    # Load model and tokenizer
    logger.info(f"Loading model and tokenizer: {args.model_name}")
    dtype = getattr(torch, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map=args.device,
        torch_dtype=dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Parse layer indices
    layer_indices = [int(x) for x in args.layers.split(",")]
    logger.info(f"Analyzing layers: {layer_indices}")
    
    # Load SAE decoder weights
    logger.info(f"Loading SAE decoder weights from source: {args.sae_source}")
    sae_loader = SAELoader(cache_dir=os.path.join(args.cache_dir, "sae"))
    
    if args.sae_source == "dummy":
        # Use dummy weights for testing
        hidden_size = model.config.hidden_size
        decoder_weights = sae_loader.load_dummy_decoder_weights(
            layer_indices=layer_indices,
            hidden_size=hidden_size,
            n_features=hidden_size * 4,  # Typical SAE has 4x features
        )
    elif args.sae_source == "gemma_scope":
        # Load from Gemma Scope (or whatever external source)
        decoder_weights = sae_loader.load_gemma_sae_decoder_weights(
            layer_indices=layer_indices,
            model_size="2b" if "2b" in args.model_name else "7b",
            source="gemma_scope",
        )
    else:
        # Load from local path
        decoder_weights = sae_loader.load_gemma_sae_decoder_weights(
            layer_indices=layer_indices,
            source=args.sae_source,
        )
    
    # Prepare dataset
    logger.info(f"Preparing dataset with {args.num_examples} examples")
    dataset = SVADataset(
        tokenizer=tokenizer,
        correct_only=args.correct_only,
        cache_dir=os.path.join(args.cache_dir, "datasets"),
    )
    splits = dataset.get_train_test_split(
        num_examples=args.num_examples,
        test_size=args.test_size,
    )
    train_dataset = splits["train"]
    test_dataset = splits["test"]
    logger.info(f"Created dataset with {len(train_dataset)} training and {len(test_dataset)} test examples")
    
    # Initialize Analytical Gradient
    logger.info("Initializing Analytical Gradient")
    analytical_gradient = AnalyticalGradient(
        model=model,
        tokenizer=tokenizer,
        sae_decoder_weights=decoder_weights,
        layer_indices=layer_indices,
        device=args.device,
        dtype=dtype,
        cache_dir=args.cache_dir,
        save_interval=args.save_interval,
    )
    
    # Run analysis on training set
    logger.info("Running analysis on training set")
    train_results = analytical_gradient.analyze_dataset(
        dataset=train_dataset,
        batch_size=args.batch_size,
        top_k=args.top_k,
        save_prefix="train_results",
    )
    
    # Run analysis on test set
    logger.info("Running analysis on test set")
    test_results = analytical_gradient.analyze_dataset(
        dataset=test_dataset,
        batch_size=args.batch_size,
        top_k=args.top_k,
        save_prefix="test_results",
    )
    
    # Save combined results
    combined_results = {
        "train_results": train_results,
        "test_results": test_results,
        "args": vars(args),
        "model_name": args.model_name,
        "num_examples": args.num_examples,
        "test_size": args.test_size,
    }
    save_results(
        combined_results,
        os.path.join(args.output_dir, args.results_file),
    )
    
    logger.info("Analysis completed")
    
    # Visualization
    results_path = os.path.join(args.output_dir, args.results_file)
    viz_output_dir = args.viz_output_dir if args.viz_output_dir else os.path.join(args.output_dir, "viz")
    log_file = os.path.join(viz_output_dir, "visualization.log")
    
    # Create output directory
    os.makedirs(viz_output_dir, exist_ok=True)
    
    # Setup logging for visualization
    setup_logging(log_file, log_level=args.log_level)
    logger.info(f"Generating visualizations from {results_path}")
    logger.info(f"Saving visualizations to {viz_output_dir}")
    
    # Load results
    try:
        with open(results_path, 'r') as f:
            results = json.load(f)
        logger.info(f"Successfully loaded results from {results_path}")
    except Exception as e:
        logger.error(f"Failed to load results from {results_path}: {e}")
        sys.exit(1)
    
    # Process metadata for dataset distribution visualization
    if 'metadata' not in results:
        results['metadata'] = {}
    
    # If we don't have num_correct and num_incorrect in metadata, try to count them
    if 'num_correct' not in results['metadata'] or 'num_incorrect' not in results['metadata']:
        # Try to get this information from the train and test datasets
        train_results = results.get('train_results', {})
        test_results = results.get('test_results', {})
        
        # For demonstration purposes, set some default values if not available
        if 'train_dataset_size' in results and 'test_dataset_size' in results:
            total_examples = results['train_dataset_size'] + results['test_dataset_size']
            # Assume 50% correct for demonstration
            results['metadata']['num_correct'] = total_examples // 2
            results['metadata']['num_incorrect'] = total_examples - (total_examples // 2)
            logger.info(f"Added estimated dataset distribution: {results['metadata']['num_correct']} correct, {results['metadata']['num_incorrect']} incorrect")
    
    # Generate visualizations
    viz_paths = {}
    
    # 1. Model Performance Graph
    logger.info("Creating model performance visualization...")
    viz_path = plot_model_performance(results, viz_output_dir)
    if viz_path:
        viz_paths['model_performance'] = viz_path
    
    # 2. Layer Importance Graph
    logger.info("Creating layer importance visualization...")
    viz_path = plot_layer_importance(results, viz_output_dir)
    if viz_path:
        viz_paths['layer_importance'] = viz_path
    
    # 3. Top Features Graph
    logger.info("Creating top features visualization...")
    viz_path = plot_top_features(results, viz_output_dir)
    if viz_path:
        viz_paths['top_features'] = viz_path
    
    # 4. Feature Importance Comparison (Train vs Test)
    logger.info("Creating feature importance comparison visualization...")
    viz_path = plot_feature_importance_comparison(results, viz_output_dir, correct_vs_incorrect=False)
    if viz_path:
        viz_paths['feature_comparison'] = viz_path
    
    # 5. Importance Distribution
    logger.info("Creating importance distribution visualization...")
    viz_path = plot_importance_distribution(results, viz_output_dir)
    if viz_path:
        viz_paths['importance_distribution'] = viz_path
    
    # 6. Dataset Distribution
    logger.info("Creating dataset distribution visualization...")
    viz_path = plot_dataset_distribution(results, viz_output_dir)
    if viz_path:
        viz_paths['dataset_distribution'] = viz_path
    
    # Save visualization paths to file for reference
    viz_index_path = os.path.join(viz_output_dir, "visualization_index.json")
    with open(viz_index_path, 'w') as f:
        json.dump(viz_paths, f, indent=2)
    
    logger.info(f"Created {len(viz_paths)} visualizations in {viz_output_dir}")
    logger.info(f"Visualization index saved to {viz_index_path}")
    
    # Print summary to console
    print(f"\nVisualization Summary:")
    print(f"----------------------")
    print(f"Generated {len(viz_paths)} visualizations:")
    
    for name, path in viz_paths.items():
        if path:
            print(f"- {name}: {path}")
    
    print(f"\nVisualization index saved to: {viz_index_path}")
    print(f"Log file: {log_file}")


if __name__ == "__main__":
    main()
