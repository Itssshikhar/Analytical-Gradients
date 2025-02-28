#!/bin/bash

# Set paths
RESULTS_DIR="/root/MATS_8/outputs/gemma-2-2b"
OUTPUT_DIR="/root/MATS_8/outputs/visualizations"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Make the Python script executable
chmod +x generate_visualizations.py

# Run the visualization generator
python generate_visualizations.py \
  --results_dir "$RESULTS_DIR" \
  --results_file "combined_results.json" \
  --output_dir "$OUTPUT_DIR"

echo "Visualizations have been generated in $OUTPUT_DIR"
echo "You can view them using your file browser or download them for viewing." 