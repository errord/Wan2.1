# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""
Migration script to convert fp32 models to bf16 safetensors format.

Usage:
    python -m wan.utils.migrate_to_bf16 --source_dir /path/to/fp32/model --target_dir /path/to/bf16/model
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Union

import torch

try:
    import safetensors.torch as safetensors
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
    print("Error: safetensors not available. Install with: pip install safetensors")
    sys.exit(1)

from .model_loader import SafetensorsModelLoader, verify_model_compatibility


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def convert_pytorch_to_safetensors(
    source_file: Path,
    target_file: Path,
    target_dtype: torch.dtype = torch.bfloat16,
    compress: bool = True
) -> bool:
    """Convert a PyTorch model file to safetensors format"""
    try:
        logging.info(f"Loading PyTorch model from {source_file}")
        
        # Load the original model
        state_dict = torch.load(source_file, map_location='cpu')
        
        # Convert to target dtype
        converted_state_dict = {}
        total_params = len(state_dict)
        
        for i, (key, tensor) in enumerate(state_dict.items()):
            if tensor.dtype.is_floating_point:
                converted_state_dict[key] = tensor.to(dtype=target_dtype)
            else:
                converted_state_dict[key] = tensor
            
            # Progress reporting
            if (i + 1) % 100 == 0 or (i + 1) == total_params:
                logging.info(f"Converted {i + 1}/{total_params} parameters")
        
        # Save as safetensors
        logging.info(f"Saving safetensors model to {target_file}")
        target_file.parent.mkdir(parents=True, exist_ok=True)
        
        safetensors.save_file(converted_state_dict, target_file)
        
        # Verify the conversion
        logging.info("Verifying conversion...")
        verify_state_dict = safetensors.load_file(target_file)
        
        if len(verify_state_dict) == len(converted_state_dict):
            logging.info("✓ Conversion successful")
            return True
        else:
            logging.error("✗ Conversion verification failed")
            return False
            
    except Exception as e:
        logging.error(f"Error converting {source_file}: {e}")
        return False


def copy_non_model_files(source_dir: Path, target_dir: Path):
    """Copy non-model files (config, tokenizer, etc.) to target directory"""
    extensions_to_copy = {'.json', '.txt', '.md', '.yml', '.yaml'}
    
    for file_path in source_dir.rglob('*'):
        if file_path.is_file() and file_path.suffix in extensions_to_copy:
            relative_path = file_path.relative_to(source_dir)
            target_path = target_dir / relative_path
            
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            import shutil
            shutil.copy2(file_path, target_path)
            logging.info(f"Copied {relative_path}")


def migrate_model_directory(
    source_dir: Union[str, Path],
    target_dir: Union[str, Path],
    target_dtype: torch.dtype = torch.bfloat16,
    overwrite: bool = False,
    verify_source: bool = True
) -> bool:
    """Migrate entire model directory from fp32 to bf16 safetensors"""
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    
    if not source_dir.exists():
        logging.error(f"Source directory does not exist: {source_dir}")
        return False
    
    if target_dir.exists() and not overwrite:
        logging.error(f"Target directory already exists: {target_dir}. Use --overwrite to overwrite.")
        return False
    
    # Verify source compatibility
    if verify_source:
        logging.info("Verifying source model compatibility...")
        compatibility = verify_model_compatibility(source_dir)
        
        if not compatibility['compatible']:
            logging.error("Source model is not compatible:")
            for issue in compatibility['issues']:
                logging.error(f"  - {issue}")
            return False
        
        logging.info(f"Source format: {compatibility['format']}")
        for rec in compatibility['recommendations']:
            logging.info(f"  - {rec}")
    
    # Find model files to convert
    pytorch_files = list(source_dir.glob("*.pth")) + list(source_dir.glob("*.bin"))
    
    if not pytorch_files:
        logging.error("No PyTorch model files found in source directory")
        return False
    
    logging.info(f"Found {len(pytorch_files)} model files to convert")
    
    # Create target directory
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert model files
    success_count = 0
    for pytorch_file in pytorch_files:
        safetensors_file = target_dir / f"{pytorch_file.stem}.safetensors"
        
        logging.info(f"Converting {pytorch_file.name} -> {safetensors_file.name}")
        
        if convert_pytorch_to_safetensors(pytorch_file, safetensors_file, target_dtype):
            success_count += 1
        else:
            logging.error(f"Failed to convert {pytorch_file.name}")
    
    # Copy non-model files
    logging.info("Copying configuration and auxiliary files...")
    copy_non_model_files(source_dir, target_dir)
    
    # Report results
    if success_count == len(pytorch_files):
        logging.info(f"✓ Successfully migrated all {success_count} model files")
        logging.info(f"✓ Migration completed: {source_dir} -> {target_dir}")
        return True
    else:
        logging.error(f"✗ Migration partially failed: {success_count}/{len(pytorch_files)} files converted")
        return False


def get_model_size_info(model_path: Path) -> dict:
    """Get information about model file size and estimated memory usage"""
    info = {}
    
    if model_path.exists():
        size_bytes = model_path.stat().st_size
        size_mb = size_bytes / (1024 * 1024)
        size_gb = size_mb / 1024
        
        info['size_bytes'] = size_bytes
        info['size_mb'] = size_mb
        info['size_gb'] = size_gb
        
        # Estimate memory savings for bf16 conversion
        if model_path.suffix in ['.pth', '.bin']:
            estimated_bf16_size = size_bytes * 0.5  # Roughly half the size
            savings_mb = (size_bytes - estimated_bf16_size) / (1024 * 1024)
            info['estimated_bf16_savings_mb'] = savings_mb
    
    return info


def main():
    parser = argparse.ArgumentParser(description="Migrate Wan models from fp32 to bf16 safetensors format")
    
    parser.add_argument("--source_dir", required=True, help="Source model directory (fp32)")
    parser.add_argument("--target_dir", required=True, help="Target model directory (bf16 safetensors)")
    parser.add_argument("--target_dtype", default="bfloat16", choices=["bfloat16", "float16"], 
                       help="Target data type for conversion")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite target directory if it exists")
    parser.add_argument("--no_verify", action="store_true", help="Skip source model verification")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--dry_run", action="store_true", help="Show what would be done without actually doing it")
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    # Convert dtype string to torch dtype
    if args.target_dtype == "bfloat16":
        target_dtype = torch.bfloat16
    elif args.target_dtype == "float16":
        target_dtype = torch.float16
    else:
        logging.error(f"Unsupported target dtype: {args.target_dtype}")
        return 1
    
    source_dir = Path(args.source_dir)
    target_dir = Path(args.target_dir)
    
    # Show model information
    logging.info(f"Source directory: {source_dir}")
    logging.info(f"Target directory: {target_dir}")
    logging.info(f"Target dtype: {target_dtype}")
    
    if args.dry_run:
        logging.info("DRY RUN - No files will be modified")
        
        # Show what would be converted
        pytorch_files = list(source_dir.glob("*.pth")) + list(source_dir.glob("*.bin"))
        
        if not pytorch_files:
            logging.info("No PyTorch model files found")
            return 0
        
        total_size = 0
        total_savings = 0
        
        for pytorch_file in pytorch_files:
            info = get_model_size_info(pytorch_file)
            total_size += info.get('size_mb', 0)
            total_savings += info.get('estimated_bf16_savings_mb', 0)
            
            logging.info(f"Would convert: {pytorch_file.name} ({info.get('size_mb', 0):.1f} MB)")
        
        logging.info(f"Total size: {total_size:.1f} MB")
        logging.info(f"Estimated savings: {total_savings:.1f} MB ({total_savings/total_size*100:.1f}%)")
        
        return 0
    
    # Perform the migration
    success = migrate_model_directory(
        source_dir=source_dir,
        target_dir=target_dir,
        target_dtype=target_dtype,
        overwrite=args.overwrite,
        verify_source=not args.no_verify
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main()) 