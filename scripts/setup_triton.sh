#!/bin/bash
# Copy exported ONNX model into the Triton model repository.
# Run this after training (which exports models/ncf.onnx).
#
# Usage: bash scripts/setup_triton.sh [onnx_path]

ONNX_PATH=${1:-models/ncf.onnx}
TRITON_MODEL_DIR="triton/ncf/1"

if [ ! -f "${ONNX_PATH}" ]; then
    echo "Error: ${ONNX_PATH} not found. Run training first."
    exit 1
fi

cp "${ONNX_PATH}" "${TRITON_MODEL_DIR}/model.onnx"
echo "Copied ${ONNX_PATH} -> ${TRITON_MODEL_DIR}/model.onnx"
