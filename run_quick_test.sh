#!/bin/bash

# This script runs a quick test of the analytical gradient approach with a smaller model
# to verify that the code works correctly before running with the full Gemma 2B model.

# Create output directory if it doesn't exist
mkdir -p outputs

echo "Running quick test with smaller model (TinyLlama)..."

python src/main.py \
  --model_name "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T" \
  --sae_source "dummy" \
  --layers "6,7,8" \
  --num_examples 50 \
  --batch_size 2 \
  --output_dir "outputs/quick_test" \
  --cache_dir "outputs/quick_test/cache" \
  --log_file "outputs/quick_test/quick_test.log" \
  --save_interval 5 \
  --top_k 10

echo "Quick test completed. Check outputs/quick_test for results." 