#!/bin/bash

# This script runs the full analytical gradient analysis with Gemma 2 2B model
# and validates the approach on subject-verb agreement tasks.

# Create output directory if it doesn't exist
mkdir -p outputs

echo "Running full analysis with Gemma 2 2B model..."

# Make sure you have enough GPU memory for this (>= 16GB recommended)
# You may need to adjust batch size based on available memory
python src/main.py \
  --model_name "google/gemma-2-2b" \
  --sae_source "gemma_scope" \
  --layers "6,7,8,9,10,11,12" \
  --num_examples 200 \
  --batch_size 1 \
  --output_dir "outputs/gemma-2-2b" \
  --cache_dir "outputs/gemma-2-2b/cache" \
  --log_file "outputs/gemma-2-2b/gemma-2-2b.log" \
  --save_interval 10 \
  --top_k 20 \
  --dtype "float16" \

echo "Full analysis completed. Check outputs/gemma-2-2b for results."

echo "To analyze results, run Jupyter notebook: notebooks/analytical_gradient_analysis.ipynb" 