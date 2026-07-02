#!/bin/bash
# Multi-GPU training script for NanoDet-Plus Barcode/QR detection.
# Usage: ./tools/train_multi_gpu.sh

# Set visible GPUs (adjust for your server)
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Optional: disable NCCL P2P if you encounter multi-GPU communication errors
# export NCCL_P2P_DISABLE=1

# Optional: reduce memory fragmentation
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

python tools/train.py config/nanodet-plus-m_480x640_barcode30k.yml
