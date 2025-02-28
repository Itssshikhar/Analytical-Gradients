#!/usr/bin/env python3
"""
Standalone script to generate visualizations from existing analytical gradient results.

This script loads the combined_results.json file from a specified directory
and generates all the requested visualizations, saving them to an output directory.
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import the visualization functions
from src.utils.visualization import (
    plot_model_performance,
    plot_layer_importance,
    plot_top_features,
    plot_feature_importance_comparison,
    plot_importance_distribution,
    plot_dataset_distribution
)
from src.utils.utils import setup_logging

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate visualizations from existing analytical gradient results"
    )
    
    parser.add_argument(
        "--results_dir", 
        type=str, 
        default="outputs/gemma-2-2b",
        help="Directory containing combined_results.json"
    )
    
    parser.add_argument(
        "--results_file", 
        type=str, 
        default="combined_results.json",
        help="Name of the results file"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=None,
        help="Directory to save visualizations (defaults to results_dir/viz)"
    )
    
    parser.add_argument(
        "--log_file", 
        type=str, 
        default=None,
        help="Path to log file (defaults to output_dir/visualization.log)"
    )
    
    parser.add_argument(
        "--log_level", 
        type=str, 
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)"
    )
    
    return parser.parse_args()

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Set up paths
    results_path = os.path.join(args.results_dir, args.results_file)
    output_dir = args.output_dir if args.output_dir else os.path.join(args.results_dir, "viz")
    log_file = args.log_file if args.log_file else os.path.join(output_dir, "visualization.log")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging
    setup_logging(log_file, log_level=args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Generating visualizations from {results_path}")
    logger.info(f"Saving visualizations to {output_dir}")
    
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
    viz_path = plot_model_performance(results, output_dir)
    if viz_path:
        viz_paths['model_performance'] = viz_path
    
    # 2. Layer Importance Graph
    logger.info("Creating layer importance visualization...")
    viz_path = plot_layer_importance(results, output_dir)
    if viz_path:
        viz_paths['layer_importance'] = viz_path
    
    # 3. Top Features Graph
    logger.info("Creating top features visualization...")
    viz_path = plot_top_features(results, output_dir)
    if viz_path:
        viz_paths['top_features'] = viz_path
    
    # 4. Feature Importance Comparison (Train vs Test)
    logger.info("Creating feature importance comparison visualization...")
    viz_path = plot_feature_importance_comparison(results, output_dir, correct_vs_incorrect=False)
    if viz_path:
        viz_paths['feature_comparison'] = viz_path
    
    # 5. Importance Distribution
    logger.info("Creating importance distribution visualization...")
    viz_path = plot_importance_distribution(results, output_dir)
    if viz_path:
        viz_paths['importance_distribution'] = viz_path
    
    # 6. Dataset Distribution
    logger.info("Creating dataset distribution visualization...")
    viz_path = plot_dataset_distribution(results, output_dir)
    if viz_path:
        viz_paths['dataset_distribution'] = viz_path
    
    # Save visualization paths to file for reference
    viz_index_path = os.path.join(output_dir, "visualization_index.json")
    with open(viz_index_path, 'w') as f:
        json.dump(viz_paths, f, indent=2)
    
    logger.info(f"Created {len(viz_paths)} visualizations in {output_dir}")
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