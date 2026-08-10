#!/bin/bash

# This script runs the Depth Anything ViT-L inference on the 'images' folder
# and saves the output to a new folder called 'images_depth'.

# Set PYTHONPATH so that the 'depth_anything_v2' module inside the 'depth' folder can be found
export PYTHONPATH=./depth

# Allow PyTorch to fall back to CPU for unsupported MPS ops (fixes upsample_bicubic2d errors)
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Run inference
# If you only want the depth prediction images (without the raw images concatenated),
# you can add the `--pred-only` flag at the end of the command below.
python depth/run.py --encoder vitl --img-path 20260530_091217_258714.jpg --outdir front_images_depth
