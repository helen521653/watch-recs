#!/bin/bash
# Convert NCF ONNX model to TensorRT engine.
# Usage: bash scripts/export_trt.sh [onnx_path] [engine_path]
#
# Requires TensorRT trtexec to be available in PATH.
# On the JupyterHub server: /usr/src/tensorrt/bin/trtexec

ONNX_PATH=${1:-models/ncf.onnx}
ENGINE_PATH=${2:-models/ncf.trt}
META_PATH="${ONNX_PATH%.onnx}.json"
NUM_ITEMS=$(python -c "import json; print(json.load(open('${META_PATH}'))['num_items'])")

trtexec \
    --onnx="${ONNX_PATH}" \
    --saveEngine="${ENGINE_PATH}" \
    --minShapes="user_ids:1,item_ids:1" \
    --optShapes="user_ids:${NUM_ITEMS},item_ids:${NUM_ITEMS}" \
    --maxShapes="user_ids:${NUM_ITEMS},item_ids:${NUM_ITEMS}" \
    --fp16

echo "TensorRT engine saved to ${ENGINE_PATH}"
