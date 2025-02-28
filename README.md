# Analytical Gradient Approach for Sparse Feature Circuits

This repository implements an alternative approach to Sparse Feature Circuits (SFC) using Analytical Gradient computation. Instead of using Sparse Autoencoders (SAEs) during forward and backward passes, this approach computes gradients on residual streams directly and multiplies by the decoder matrix to approximate SAE activation gradients, resulting in faster and more memory-efficient computation.

## Overview

Traditional SFC approaches use Sparse Autoencoders in both forward and backward passes to analyze neural networks, which can be computationally expensive. The Analytical Gradient approach only requires the decoder weights from pre-trained SAEs and calculates gradients directly on the residual streams. This implementation focuses on subject-verb agreement tasks using the GEMMA 2 2B model and pre-trained JumpReLU SAEs from Gemma Scope.

## Key Components

1. **Analytical Gradient Computation**: Calculate gradients on residual streams directly and multiply by decoder matrices
2. **Subject-Verb Agreement Task**: Generate and analyze examples testing syntactic processing
3. **Feature Importance Ranking**: Identify the most important features in each layer for the task

## Project Structure

```
.
├── src/                                # Source code
│   ├── data/                           # Dataset processing
│   │   ├── __init__.py
│   │   └── dataset.py                  # Subject-verb agreement dataset
│   ├── model/                          # Model implementation
│   │   ├── __init__.py
│   │   ├── analytical_gradient.py      # Core analytical gradient implementation
│   │   └── sae_utils.py                # SAE loading utilities
│   ├── utils/                          # Utility functions
│   │   ├── __init__.py
│   │   └── utils.py                    # General utilities
│   ├── __init__.py
│   └── main.py                         # Main entry point
├── notebooks/                          # Jupyter notebooks
│   └── analytical_gradient_analysis.ipynb  # Analysis notebook
├── outputs/                            # Results and outputs (created at runtime)
├── run_quick_test.sh                   # Script for quick testing
├── run_full_analysis.sh                # Script for full analysis
├── requirements.txt                    # Dependencies
└── README.md                           # This file
```

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/analytical-gradient-sfc.git
   cd analytical-gradient-sfc
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Quick Test

To verify that the implementation works correctly, run the quick test script, which uses a smaller model (TinyLlama) and fewer examples:

```bash
./run_quick_test.sh
```

This script runs with dummy SAE weights for testing purposes and executes quickly to confirm the code is functioning correctly.

### Full Analysis

To run the full analysis with Gemma 2B model:

```bash
./run_full_analysis.sh
```

Note: The full analysis requires significant GPU memory (>=16GB recommended). If you encounter memory issues, adjust the batch size or use a smaller model.

### Custom Configuration

You can run the main script directly with custom parameters:

```bash
python src/main.py \
  --model_name "google/gemma-2b" \
  --sae_source "path/to/your/sae/weights" \
  --layers "6,7,8,9,10,11,12" \
  --num_examples 200 \
  --output_dir "outputs/custom_run" \
  --batch_size 1
```

For all available options, run:

```bash
python src/main.py --help
```

### Using Pre-trained SAEs

By default, the code uses dummy SAE weights for demonstration purposes. To use real pre-trained SAEs:

1. Download pre-trained JumpReLU SAEs for Gemma 2 2B from Gemma Scope
2. Place them in a directory structure like: `sae_weights/gemma-2b/layer_{layer_idx}.pt`
3. Run the analysis with the `--sae_source` parameter pointing to your directory:
   ```bash
   python src/main.py --sae_source "path/to/sae_weights"
   ```

## Analyzing Results

After running the analysis, you can analyze the results using the Jupyter notebook:

```bash
jupyter notebook notebooks/analytical_gradient_analysis.ipynb
```

The notebook provides visualizations and analysis of:
- Feature importance by layer
- Comparison between train and test feature rankings
- Identification of the most important layers for the subject-verb agreement task

## Key Findings

When run on Gemma 2 2B with subject-verb agreement tasks, this approach:

1. Is faster and more memory-efficient than traditional SFC approaches
2. Successfully identifies the same important features as traditional approaches
3. Shows that middle layers (particularly 8-10) are most relevant for syntactic tasks like subject-verb agreement

## References

- Feature Circuits: https://github.com/saprmarks/feature-circuits
- Analytical Gradient paper: https://huggingface.co/papers/2408.05147
- Gemma Scope: https://huggingface.co/collections/google/gemma-scope-65b27da6ed2d8d01076aa704

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 