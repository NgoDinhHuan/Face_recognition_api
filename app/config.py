# config.py
import os

# Model paths
MODEL_TYPE = "fp16"  # Default model type
MODEL_PATH = f"models/edgeface_{MODEL_TYPE}.onnx"

# ONNX Runtime Providers
PREFERRED_PROVIDERS = ['CUDAExecutionProvider', 'CPUExecutionProvider']
