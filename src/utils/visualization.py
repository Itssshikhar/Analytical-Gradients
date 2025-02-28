"""
Visualization utilities for the Analytical Gradient approach.

This module provides functions to create various visualizations for
analyzing and presenting results from the Analytical Gradient analysis.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List, Optional, Any, Union
import logging
import json

logger = logging.getLogger(__name__)

def set_plot_style():
    """Set consistent style for all plots."""
    sns.set_theme(style="whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['figure.titlesize'] = 18


def plot_model_performance(results: Dict[str, Any], output_dir: str) -> str:
    """
    Create a bar chart showing the model's average probability of the actual verb
    for correct and incorrect sentences.
    
    Args:
        results: Dictionary containing results from analytical gradient analysis
        output_dir: Directory to save the visualization
        
    Returns:
        Path to the saved visualization
    """
    set_plot_style()
    
    # Extract probabilities for correct and incorrect examples
    train_data = results.get('train_results', {})
    test_data = results.get('test_results', {})
    all_data = {}
    
    # Combine data
    for dataset_name, dataset in [('train', train_data), ('test', test_data)]:
        if 'metadata' in dataset and 'probabilities' in dataset['metadata']:
            probs = dataset['metadata']['probabilities']
            if 'correct_examples_probs' in probs and 'incorrect_examples_probs' in probs:
                all_data[f"{dataset_name}_correct"] = np.mean(probs['correct_examples_probs'])
                all_data[f"{dataset_name}_incorrect"] = np.mean(probs['incorrect_examples_probs'])
    
    # If metadata doesn't have probabilities field, check if we have it at the top level
    if not all_data and 'probabilities' in results:
        probs = results['probabilities']
        if 'correct_examples_probs' in probs and 'incorrect_examples_probs' in probs:
            all_data["correct"] = np.mean(probs['correct_examples_probs'])
            all_data["incorrect"] = np.mean(probs['incorrect_examples_probs'])
    
    # If we don't have any data, return early
    if not all_data:
        logger.warning("No probability data found in results. Cannot create model performance graph.")
        return ""
    
    # Create dataframe for plotting
    df = pd.DataFrame({
        'Sentence Type': list(all_data.keys()),
        'Average Probability': list(all_data.values())
    })
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='Sentence Type', y='Average Probability', data=df)
    
    # Add value labels on top of bars
    for i, p in enumerate(ax.patches):
        ax.annotate(f'{p.get_height():.3f}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom')
    
    plt.title("Model's Performance on Subject-Verb Agreement Task")
    plt.ylabel("Average Probability of the Actual Verb")
    plt.xlabel("Type of Sentence")
    plt.ylim(0, 1.1)  # Probabilities are between 0 and 1
    plt.tight_layout()
    
    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "model_performance.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Model performance graph saved to {output_path}")
    return output_path


def plot_layer_importance(results: Dict[str, Any], output_dir: str) -> str:
    """
    Create a bar chart showing the average feature importance per layer.
    
    Args:
        results: Dictionary containing results from analytical gradient analysis
        output_dir: Directory to save the visualization
        
    Returns:
        Path to the saved visualization
    """
    set_plot_style()
    
    layer_importance = {}
    
    # Process both train and test results
    for dataset_name, dataset in [('train', results.get('train_results', {})), 
                                 ('test', results.get('test_results', {}))]:
        # Skip metadata entries
        for key, value in dataset.items():
            if key.startswith('layer_'):
                # Extract layer number
                layer_num = int(key.split('_')[1])
                
                # Calculate average importance (mean of top feature values)
                if 'top_feature_values' in value:
                    avg_importance = np.mean(value['top_feature_values'])
                    
                    # Add to layer_importance dict
                    if layer_num not in layer_importance:
                        layer_importance[layer_num] = {}
                    layer_importance[layer_num][dataset_name] = avg_importance
    
    # Create dataframe for plotting
    rows = []
    for layer, importances in layer_importance.items():
        for dataset, value in importances.items():
            rows.append({
                'Layer': f"Layer {layer}",
                'Average Importance': value,
                'Dataset': dataset
            })
    
    df = pd.DataFrame(rows)
    
    # If we have both train and test data, create a grouped bar chart
    if 'train' in df['Dataset'].unique() and 'test' in df['Dataset'].unique():
        plt.figure(figsize=(12, 8))
        ax = sns.barplot(x='Layer', y='Average Importance', hue='Dataset', data=df)
    else:
        # Otherwise create a simple bar chart
        plt.figure(figsize=(12, 8))
        df_sorted = df.sort_values('Layer')
        ax = sns.barplot(x='Layer', y='Average Importance', data=df_sorted)
    
    plt.title("Feature Importance by Layer")
    plt.ylabel("Average Feature Importance Score")
    plt.xlabel("Layer")
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "layer_importance.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Layer importance graph saved to {output_path}")
    return output_path


def plot_top_features(results: Dict[str, Any], output_dir: str, layer: Optional[int] = None, top_n: int = 10) -> str:
    """
    Create a bar chart showing the top k features and their importance scores.
    If layer is None, select the layer with the highest average importance.
    
    Args:
        results: Dictionary containing results from analytical gradient analysis
        output_dir: Directory to save the visualization
        layer: Specific layer to visualize (if None, use the most important layer)
        top_n: Number of top features to display
        
    Returns:
        Path to the saved visualization
    """
    set_plot_style()
    
    # Use training results for visualization
    train_results = results.get('train_results', {})
    
    # If layer is not specified, find the layer with highest average importance
    if layer is None:
        max_importance = -1
        for key, value in train_results.items():
            if key.startswith('layer_') and 'top_feature_values' in value:
                avg_importance = np.mean(value['top_feature_values'])
                if avg_importance > max_importance:
                    max_importance = avg_importance
                    layer = int(key.split('_')[1])
    
    if layer is None:
        logger.warning("No layer data found in results. Cannot create top features graph.")
        return ""
    
    # Get the data for the selected layer
    layer_key = f'layer_{layer}'
    if layer_key not in train_results:
        logger.warning(f"Layer {layer} not found in results. Cannot create top features graph.")
        return ""
    
    layer_data = train_results[layer_key]
    
    # Get indices and values of top features
    indices = layer_data.get('top_feature_indices', [])
    values = layer_data.get('top_feature_values', [])
    
    # Ensure we don't try to plot more features than available
    top_n = min(top_n, len(indices))
    
    # Create dataframe for plotting
    df = pd.DataFrame({
        'Feature Index': [str(idx) for idx in indices[:top_n]],
        'Importance Score': values[:top_n]
    })
    
    # Sort by importance score
    df = df.sort_values('Importance Score', ascending=False)
    
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x='Feature Index', y='Importance Score', data=df)
    
    # Add value labels on top of bars
    for i, p in enumerate(ax.patches):
        ax.annotate(f'{p.get_height():.4f}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom')
    
    plt.title(f"Top {top_n} Features by Importance Score in Layer {layer}")
    plt.ylabel("Importance Score")
    plt.xlabel("Feature Index")
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"top_features_layer_{layer}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Top features graph for layer {layer} saved to {output_path}")
    return output_path


def plot_feature_importance_comparison(results: Dict[str, Any], output_dir: str, 
                                      correct_vs_incorrect: bool = True) -> str:
    """
    Create a bar chart comparing feature importance for correct vs. incorrect instances.
    
    Args:
        results: Dictionary containing results from analytical gradient analysis
        output_dir: Directory to save the visualization
        correct_vs_incorrect: If True, compare correct vs incorrect, else train vs test
        
    Returns:
        Path to the saved visualization
    """
    set_plot_style()
    
    if correct_vs_incorrect:
        # For this we need the metadata with correct vs incorrect classifications
        if ('metadata' not in results or 
            'correct_instances' not in results['metadata'] or 
            'incorrect_instances' not in results['metadata']):
            logger.warning("No correct/incorrect classification data found. Cannot create comparison graph.")
            return ""
        
        # Extract feature importance for correct and incorrect instances
        # This implementation depends on how the data is structured in results
        # Placeholder implementation:
        importance_data = {
            'Correct': results['metadata'].get('correct_feature_importance', {}),
            'Incorrect': results['metadata'].get('incorrect_feature_importance', {})
        }
    else:
        # Compare train vs test
        train_data = results.get('train_results', {})
        test_data = results.get('test_results', {})
        
        # Find common layers
        train_layers = [k for k in train_data.keys() if k.startswith('layer_')]
        test_layers = [k for k in test_data.keys() if k.startswith('layer_')]
        common_layers = set(train_layers).intersection(set(test_layers))
        
        if not common_layers:
            logger.warning("No common layers found between train and test data. Cannot create comparison graph.")
            return ""
        
        # Select the layer with highest importance in train data
        selected_layer = None
        max_importance = -1
        for layer_key in common_layers:
            if 'top_feature_values' in train_data[layer_key]:
                avg_importance = np.mean(train_data[layer_key]['top_feature_values'])
                if avg_importance > max_importance:
                    max_importance = avg_importance
                    selected_layer = layer_key
        
        if selected_layer is None:
            logger.warning("Could not find a suitable layer for comparison. Cannot create comparison graph.")
            return ""
        
        # Get top features and their importance scores
        train_indices = train_data[selected_layer].get('top_feature_indices', [])
        train_values = train_data[selected_layer].get('top_feature_values', [])
        test_indices = test_data[selected_layer].get('top_feature_indices', [])
        test_values = test_data[selected_layer].get('top_feature_values', [])
        
        # Create dataframes
        train_df = pd.DataFrame({
            'Feature Index': train_indices,
            'Importance': train_values,
            'Dataset': 'Train'
        })
        
        test_df = pd.DataFrame({
            'Feature Index': test_indices,
            'Importance': test_values,
            'Dataset': 'Test'
        })
        
        # Combine dataframes
        df = pd.concat([train_df, test_df])
        
        # Get common top features
        common_features = set(train_indices).intersection(set(test_indices))
        df_common = df[df['Feature Index'].isin(common_features)]
        
        plt.figure(figsize=(14, 8))
        ax = sns.barplot(x='Feature Index', y='Importance', hue='Dataset', data=df_common)
        
        plt.title(f"Feature Importance Comparison (Train vs Test) for {selected_layer}")
        plt.ylabel("Importance Score")
        plt.xlabel("Feature Index")
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save the figure
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "feature_importance_train_vs_test.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Feature importance comparison graph saved to {output_path}")
        return output_path


def plot_importance_distribution(results: Dict[str, Any], output_dir: str) -> str:
    """
    Create a histogram showing the distribution of importance scores across all features.
    
    Args:
        results: Dictionary containing results from analytical gradient analysis
        output_dir: Directory to save the visualization
        
    Returns:
        Path to the saved visualization
    """
    set_plot_style()
    
    # Use training results for visualization
    train_results = results.get('train_results', {})
    
    # Collect all feature importance values across layers
    all_values = []
    layer_values = {}
    
    for key, value in train_results.items():
        if key.startswith('layer_') and 'top_feature_values' in value:
            layer_num = int(key.split('_')[1])
            values = value['top_feature_values']
            all_values.extend(values)
            layer_values[layer_num] = values
    
    if not all_values:
        logger.warning("No feature importance data found. Cannot create distribution graph.")
        return ""
    
    plt.figure(figsize=(12, 8))
    
    # Create histogram with KDE
    sns.histplot(all_values, kde=True, bins=30)
    
    plt.title("Distribution of Feature Importance Scores")
    plt.xlabel("Importance Score")
    plt.ylabel("Frequency")
    plt.tight_layout()
    
    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "importance_distribution.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Importance distribution graph saved to {output_path}")
    
    # Also create per-layer distributions
    plt.figure(figsize=(14, 10))
    
    for layer, values in layer_values.items():
        sns.kdeplot(values, label=f"Layer {layer}")
    
    plt.title("Distribution of Feature Importance Scores by Layer")
    plt.xlabel("Importance Score")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    
    layer_output_path = os.path.join(output_dir, "importance_distribution_by_layer.png")
    plt.savefig(layer_output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Per-layer importance distribution graph saved to {layer_output_path}")
    
    return output_path


def plot_dataset_distribution(results: Dict[str, Any], output_dir: str) -> str:
    """
    Create a pie chart showing the distribution of correct and incorrect sentences in the dataset.
    
    Args:
        results: Dictionary containing results from analytical gradient analysis
        output_dir: Directory to save the visualization
        
    Returns:
        Path to the saved visualization
    """
    set_plot_style()
    
    # Extract dataset information if available
    metadata = results.get('metadata', {})
    
    # Check if we have the necessary data
    if 'num_correct' not in metadata or 'num_incorrect' not in metadata:
        logger.warning("Dataset distribution data not found. Cannot create pie chart.")
        return ""
    
    num_correct = metadata['num_correct']
    num_incorrect = metadata['num_incorrect']
    
    # Create pie chart
    plt.figure(figsize=(10, 8))
    
    labels = ['Correct Sentences', 'Incorrect Sentences']
    sizes = [num_correct, num_incorrect]
    colors = ['#66b3ff', '#ff9999']
    explode = (0.1, 0)  # explode the 1st slice for emphasis
    
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=140)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    
    plt.title("Dataset Distribution: Correct vs Incorrect Sentences")
    plt.tight_layout()
    
    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "dataset_distribution.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Dataset distribution graph saved to {output_path}")
    return output_path


def generate_all_visualizations(results_path: str, output_dir: str) -> Dict[str, str]:
    """
    Generate all visualizations from results file.
    
    Args:
        results_path: Path to the results JSON file
        output_dir: Directory to save visualizations
        
    Returns:
        Dictionary mapping visualization names to their file paths
    """
    # Load results
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Generate visualizations
    visualization_paths = {}
    
    # 1. Model Performance Graph
    viz_path = plot_model_performance(results, output_dir)
    if viz_path:
        visualization_paths['model_performance'] = viz_path
    
    # 2. Layer Importance Graph
    viz_path = plot_layer_importance(results, output_dir)
    if viz_path:
        visualization_paths['layer_importance'] = viz_path
    
    # 3. Top Features Graph (for most important layer)
    viz_path = plot_top_features(results, output_dir)
    if viz_path:
        visualization_paths['top_features'] = viz_path
    
    # 4. Feature Importance Comparison (Train vs Test)
    viz_path = plot_feature_importance_comparison(results, output_dir, correct_vs_incorrect=False)
    if viz_path:
        visualization_paths['feature_importance_comparison'] = viz_path
    
    # 5. Importance Distribution
    viz_path = plot_importance_distribution(results, output_dir)
    if viz_path:
        visualization_paths['importance_distribution'] = viz_path
    
    # 6. Dataset Distribution
    viz_path = plot_dataset_distribution(results, output_dir)
    if viz_path:
        visualization_paths['dataset_distribution'] = viz_path
    
    return visualization_paths 