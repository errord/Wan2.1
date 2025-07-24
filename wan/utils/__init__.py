# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.

from .model_loader import (
    SafetensorsModelLoader,
    ModelInitializer, 
    patch_wan_models,
    create_bf16_safetensors_config,
    verify_model_compatibility
)

# Auto-patch models on import
try:
    patch_wan_models()
except Exception as e:
    import logging
    logging.warning(f"Could not auto-patch model classes: {e}")

__all__ = [
    'SafetensorsModelLoader',
    'ModelInitializer',
    'patch_wan_models', 
    'create_bf16_safetensors_config',
    'verify_model_compatibility'
]
