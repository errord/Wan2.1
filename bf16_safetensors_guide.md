# BF16 Safetensors Model Support Guide

This guide explains how to use bf16 quantized safetensors models with the Wan2.1 framework for improved memory efficiency and performance.

## Overview

The enhanced model loader supports:
- **Safetensors format**: Fast, secure model serialization
- **BF16 quantization**: Half memory usage compared to FP32
- **Automatic format detection**: Seamlessly handles both formats
- **Memory-efficient loading**: Optimized for large models

## Benefits

- **50% memory reduction**: BF16 uses half the memory of FP32
- **Faster loading**: Safetensors loads significantly faster than PyTorch checkpoints
- **Better compatibility**: Safetensors is cross-platform and framework agnostic
- **Automatic fallback**: Falls back to original loader if enhanced loader fails

## Quick Start

### 1. Install Dependencies

```bash
pip install safetensors
```

### 2. Enable Enhanced Loading

The enhanced loader is automatically enabled when you update your configuration:

```python
# Your config will automatically use enhanced loading
config.use_enhanced_loader = True  # Already set in shared_config.py
```

### 3. Use with Existing Code

No code changes needed! The enhanced loader works automatically:

```python
import wan

# This will automatically use bf16 safetensors if available
model = wan.WanT2V(
    config=config,
    checkpoint_dir="path/to/your/model"
)
```

## Migration from FP32 to BF16

### Option 1: Use Migration Script

Convert your existing FP32 models to BF16 safetensors:

```bash
# Dry run to see what will be converted
python -m wan.utils.migrate_to_bf16 \
    --source_dir /path/to/fp32/model \
    --target_dir /path/to/bf16/model \
    --dry_run

# Perform the actual conversion
python -m wan.utils.migrate_to_bf16 \
    --source_dir /path/to/fp32/model \
    --target_dir /path/to/bf16/model \
    --overwrite
```

### Option 2: Manual Conversion

```python
from wan.utils.model_loader import SafetensorsModelLoader
import torch

# Convert PyTorch checkpoint to BF16 safetensors
loader = SafetensorsModelLoader()
state_dict = loader.load_pytorch_state_dict(
    "model.pth", 
    target_dtype=torch.bfloat16
)

import safetensors.torch as safetensors
safetensors.save_file(state_dict, "model.safetensors")
```

## Advanced Usage

### Custom Model Loading

```python
from wan.modules.model import WanModel

# Direct enhanced loading
model = WanModel.from_pretrained_bf16(
    checkpoint_dir="path/to/model",
    device="cpu"
)

# Auto-detection with fallback
model = WanModel.auto_from_pretrained(
    checkpoint_dir="path/to/model"
)
```

### Configuration Options

```python
from wan.utils.model_loader import create_bf16_safetensors_config

# Get optimal configuration
config = create_bf16_safetensors_config()
print(config)
# {
#     'use_safetensors': True,
#     'target_dtype': torch.bfloat16,
#     'model_filename': None,  # Auto-detect
#     'device': 'cpu',  # Load to CPU first
#     'memory_efficient': True,
#     'auto_detect_dtype': True
# }
```

### Model Compatibility Check

```python
from wan.utils.model_loader import verify_model_compatibility

# Check if your model is compatible
result = verify_model_compatibility("path/to/model")
print(f"Compatible: {result['compatible']}")
print(f"Format: {result['format']}")
for rec in result['recommendations']:
    print(f"- {rec}")
```

## Configuration Reference

### Shared Configuration

The following options are available in `wan_shared_cfg`:

```python
# Model loading configuration
wan_shared_cfg.use_enhanced_loader = True
wan_shared_cfg.prefer_safetensors = True
wan_shared_cfg.auto_detect_dtype = True
wan_shared_cfg.memory_efficient_loading = True

# BF16 safetensors specific config
wan_shared_cfg.bf16_safetensors_config = {
    'target_dtype': torch.bfloat16,
    'device': "cpu",
    'memory_efficient': True,
    'auto_detect_dtype': True
}
```

### Per-Model Configuration

You can override settings for specific models:

```python
# Disable enhanced loading for a specific model
model = wan.WanT2V(
    config=config,
    checkpoint_dir="path/to/model",
    use_enhanced_loader=False  # Force standard loading
)
```

## Troubleshooting

### Common Issues

1. **"safetensors not available" error**
   ```bash
   pip install safetensors
   ```

2. **"Enhanced loader not available" warning**
   - This is normal fallback behavior
   - The standard loader will be used instead

3. **Memory issues during conversion**
   ```python
   # Use memory-efficient conversion
   loader.enhanced_from_pretrained(
       model_class, checkpoint_dir,
       memory_efficient=True
   )
   ```

4. **Missing model files**
   - Check that model files exist in checkpoint directory
   - Use `verify_model_compatibility()` to diagnose issues

### Performance Tips

1. **Use CPU for initial loading**
   ```python
   model = WanModel.from_pretrained_bf16(
       checkpoint_dir="path/to/model",
       device="cpu"  # Load to CPU first
   )
   model.to("cuda")  # Move to GPU after loading
   ```

2. **Enable memory-efficient loading**
   ```python
   # Automatically enabled in enhanced loader
   memory_efficient=True
   ```

3. **Prefer safetensors format**
   - Convert models to safetensors for fastest loading
   - Use the migration script for batch conversion

## API Reference

### SafetensorsModelLoader

Main class for enhanced model loading:

```python
class SafetensorsModelLoader:
    @staticmethod
    def detect_model_format(checkpoint_dir) -> str
    
    @staticmethod
    def load_safetensors_state_dict(checkpoint_path, target_dtype=torch.bfloat16, device="cpu", memory_efficient=True) -> dict
    
    @staticmethod
    def load_pytorch_state_dict(checkpoint_path, target_dtype=torch.bfloat16, device="cpu", memory_efficient=True) -> dict
    
    @classmethod
    def enhanced_from_pretrained(cls, model_class, checkpoint_dir, model_filename=None, target_dtype=torch.bfloat16, device="cpu", memory_efficient=True, auto_detect_dtype=True, **kwargs)
```

### ModelInitializer

Automatic model initialization wrapper:

```python
class ModelInitializer:
    @staticmethod
    def auto_initialize_model(model_class, checkpoint_dir, prefer_safetensors=True, target_dtype=torch.bfloat16, device="cpu", **kwargs)
```

### Enhanced Model Methods

Added to WanModel and VaceWanModel:

```python
# BF16 safetensors loading
model = WanModel.from_pretrained_bf16(checkpoint_dir, device="cpu")

# Auto-detection with fallback
model = WanModel.auto_from_pretrained(checkpoint_dir)

# Enhanced loading (via patching)
model = WanModel.from_pretrained_enhanced(checkpoint_dir, target_dtype=torch.bfloat16)
```

## Examples

### Complete T2V Example

```python
import wan
from wan.configs import WAN_CONFIGS

# Configuration automatically uses enhanced loading
config = WAN_CONFIGS['t2v-14B']

# Model will automatically use bf16 safetensors if available
model = wan.WanT2V(
    config=config,
    checkpoint_dir="path/to/bf16/model",
    device_id=0
)

# Generate video
video = model.generate(
    prompt="A cat playing with a ball",
    size=(1280, 720),
    frame_num=81
)
```

### Model Migration Example

```bash
# Check what will be converted
python -m wan.utils.migrate_to_bf16 \
    --source_dir ./Wan2.1-T2V-14B \
    --target_dir ./Wan2.1-T2V-14B-BF16 \
    --dry_run \
    --verbose

# Perform conversion
python -m wan.utils.migrate_to_bf16 \
    --source_dir ./Wan2.1-T2V-14B \
    --target_dir ./Wan2.1-T2V-14B-BF16 \
    --target_dtype bfloat16 \
    --overwrite
```

### Compatibility Check Example

```python
from wan.utils.model_loader import verify_model_compatibility

# Check model compatibility
result = verify_model_compatibility("./my-model")

if result['compatible']:
    print(f"✓ Model is compatible (format: {result['format']})")
    for rec in result['recommendations']:
        print(f"  • {rec}")
else:
    print("✗ Model is not compatible:")
    for issue in result['issues']:
        print(f"  • {issue}")
```

## Performance Comparison

| Model Size | FP32 Memory | BF16 Memory | Memory Savings | Loading Speed |
|------------|-------------|-------------|----------------|---------------|
| 1.3B       | ~5.2 GB     | ~2.6 GB     | 50%            | 2-3x faster   |
| 14B        | ~56 GB      | ~28 GB      | 50%            | 2-3x faster   |

## Migration Checklist

- [ ] Install safetensors: `pip install safetensors`
- [ ] Backup original models
- [ ] Run migration script with `--dry_run` first
- [ ] Convert models: `python -m wan.utils.migrate_to_bf16`
- [ ] Test converted models
- [ ] Update model paths in your code
- [ ] Verify memory usage and performance improvements

## Support

For issues related to bf16 safetensors support:

1. Check model compatibility with `verify_model_compatibility()`
2. Try migration script with `--verbose` and `--dry_run`
3. Enable detailed logging in your application
4. Check system requirements (GPU memory, disk space)

The enhanced loader provides automatic fallback to ensure compatibility with existing workflows while offering improved performance when possible. 