# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import os
import logging
import torch
import gc
from typing import Optional, Union, Dict, Any
from pathlib import Path

try:
    import safetensors.torch as safetensors
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
    logging.warning("safetensors not available. Install with: pip install safetensors")

from ..modules.model import WanModel
from ..modules.vace_model import VaceWanModel


class SafetensorsModelLoader:
    """Enhanced model loader supporting safetensors format and bf16 quantization"""
    
    @staticmethod
    def detect_model_format(checkpoint_dir: Union[str, Path]) -> str:
        """Detect the model checkpoint format in the directory"""
        checkpoint_dir = Path(checkpoint_dir)
        
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_dir}")
        
        # Check for safetensors files
        safetensors_files = list(checkpoint_dir.glob("*.safetensors"))
        if safetensors_files:
            logging.info(f"Found {len(safetensors_files)} safetensors files")
            return "safetensors"
        
        # Check for pytorch files
        pytorch_files = list(checkpoint_dir.glob("*.pth")) + list(checkpoint_dir.glob("*.bin"))
        if pytorch_files:
            logging.info(f"Found {len(pytorch_files)} pytorch files")
            return "pytorch"
        
        raise FileNotFoundError(f"No supported model files found in {checkpoint_dir}")
    
    @staticmethod
    def get_model_info(checkpoint_path: Union[str, Path]) -> Dict[str, Any]:
        """Get information about the model file"""
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Model file does not exist: {checkpoint_path}")
        
        info = {
            "path": str(checkpoint_path),
            "size_mb": checkpoint_path.stat().st_size / (1024 * 1024),
            "format": "safetensors" if checkpoint_path.suffix == ".safetensors" else "pytorch"
        }
        
        if checkpoint_path.suffix == ".safetensors" and SAFETENSORS_AVAILABLE:
            try:
                # Get metadata without loading the full model
                with safetensors.safe_open(checkpoint_path, framework="pt") as f:
                    info["keys"] = list(f.keys())
                    info["num_parameters"] = len(info["keys"])
                    
                    # Get dtype of first parameter
                    first_key = info["keys"][0] if info["keys"] else None
                    if first_key:
                        tensor = f.get_tensor(first_key)
                        info["dtype"] = str(tensor.dtype)
                        info["device"] = str(tensor.device)
            except Exception as e:
                logging.warning(f"Could not read safetensors metadata: {e}")
        
        return info
    
    @staticmethod
    def load_safetensors_state_dict(
        checkpoint_path: Union[str, Path],
        target_dtype: torch.dtype = torch.bfloat16,
        device: str = "cpu",
        memory_efficient: bool = True
    ) -> dict:
        """Load state dict from safetensors file with dtype conversion and memory management"""
        if not SAFETENSORS_AVAILABLE:
            raise ImportError("safetensors is required for loading .safetensors files")
        
        checkpoint_path = Path(checkpoint_path)
        logging.info(f"Loading safetensors model from {checkpoint_path}")
        
        # Clear GPU cache before loading
        if memory_efficient and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        try:
            # Load safetensors with device mapping
            state_dict = safetensors.load_file(str(checkpoint_path), device=device)
            
            # Convert to target dtype
            if target_dtype != torch.float32:
                logging.info(f"Converting model weights to {target_dtype}")
                converted_state_dict = {}
                
                for key, tensor in state_dict.items():
                    if tensor.dtype.is_floating_point:
                        converted_state_dict[key] = tensor.to(dtype=target_dtype)
                    else:
                        converted_state_dict[key] = tensor
                    
                    # Memory management for large models
                    if memory_efficient and len(converted_state_dict) % 100 == 0:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                
                # Clean up original state_dict
                del state_dict
                if memory_efficient:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                state_dict = converted_state_dict
            
            logging.info(f"Successfully loaded safetensors model with {len(state_dict)} parameters")
            return state_dict
            
        except Exception as e:
            logging.error(f"Error loading safetensors file {checkpoint_path}: {e}")
            raise
    
    @staticmethod
    def load_pytorch_state_dict(
        checkpoint_path: Union[str, Path],
        target_dtype: torch.dtype = torch.bfloat16,
        device: str = "cpu",
        memory_efficient: bool = True
    ) -> dict:
        """Load state dict from pytorch file with dtype conversion and memory management"""
        checkpoint_path = Path(checkpoint_path)
        logging.info(f"Loading pytorch model from {checkpoint_path}")
        
        # Clear GPU cache before loading
        if memory_efficient and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        try:
            state_dict = torch.load(str(checkpoint_path), map_location=device)
            
            # Convert to target dtype
            if target_dtype != torch.float32:
                logging.info(f"Converting model weights to {target_dtype}")
                converted_state_dict = {}
                
                for key, tensor in state_dict.items():
                    if tensor.dtype.is_floating_point:
                        converted_state_dict[key] = tensor.to(dtype=target_dtype)
                    else:
                        converted_state_dict[key] = tensor
                    
                    # Memory management for large models
                    if memory_efficient and len(converted_state_dict) % 100 == 0:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                
                # Clean up original state_dict
                del state_dict
                if memory_efficient:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                state_dict = converted_state_dict
            
            logging.info(f"Successfully loaded pytorch model with {len(state_dict)} parameters")
            return state_dict
            
        except Exception as e:
            logging.error(f"Error loading pytorch file {checkpoint_path}: {e}")
            raise
    
    @classmethod
    def enhanced_from_pretrained(
        cls,
        model_class,
        checkpoint_dir: Union[str, Path],
        model_filename: Optional[str] = None,
        target_dtype: torch.dtype = torch.bfloat16,
        device: str = "cpu",
        memory_efficient: bool = True,
        auto_detect_dtype: bool = True,
        **kwargs
    ):
        """Enhanced from_pretrained method supporting safetensors and dtype conversion"""
        checkpoint_dir = Path(checkpoint_dir)
        
        # Detect model format if filename not specified
        if model_filename is None:
            format_type = cls.detect_model_format(checkpoint_dir)
            
            if format_type == "safetensors":
                # Look for the main model file
                safetensors_files = list(checkpoint_dir.glob("*.safetensors"))
                # Prefer model files with 'model' in the name
                model_files = [f for f in safetensors_files if 'model' in f.name.lower()]
                if model_files:
                    model_filename = model_files[0].name
                else:
                    model_filename = safetensors_files[0].name
            else:
                # Look for pytorch files
                pytorch_files = list(checkpoint_dir.glob("*.pth")) + list(checkpoint_dir.glob("*.bin"))
                # Prefer model files with 'model' in the name
                model_files = [f for f in pytorch_files if 'model' in f.name.lower()]
                if model_files:
                    model_filename = model_files[0].name
                else:
                    model_filename = pytorch_files[0].name
        
        checkpoint_path = checkpoint_dir / model_filename
        
        # Get model info for logging
        try:
            model_info = cls.get_model_info(checkpoint_path)
            logging.info(f"Model info: {model_info}")
            
            # Auto-detect dtype from model if requested
            if auto_detect_dtype and "dtype" in model_info:
                detected_dtype = model_info["dtype"]
                if "bfloat16" in detected_dtype and target_dtype == torch.bfloat16:
                    logging.info("Model already in bfloat16 format, using direct loading")
                elif "float16" in detected_dtype:
                    logging.info("Model in float16 format, converting to bfloat16")
                elif "float32" in detected_dtype:
                    logging.info("Model in float32 format, converting to bfloat16")
        except Exception as e:
            logging.warning(f"Could not get model info: {e}")
        
        # Determine loading method based on file extension
        if checkpoint_path.suffix == ".safetensors":
            state_dict = cls.load_safetensors_state_dict(
                checkpoint_path, target_dtype, device, memory_efficient
            )
        else:
            state_dict = cls.load_pytorch_state_dict(
                checkpoint_path, target_dtype, device, memory_efficient
            )
        
        # Create model instance with target dtype
        logging.info(f"Creating model instance with device: {device}")
        with torch.device(device):
            model = model_class(**kwargs)
        
        # Load state dict
        logging.info("Loading state dict into model...")
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            logging.warning(f"Missing keys when loading model: {missing_keys[:10]}...")  # Show first 10
        if unexpected_keys:
            logging.warning(f"Unexpected keys when loading model: {unexpected_keys[:10]}...")  # Show first 10
        
        # Convert model to target dtype
        model = model.to(dtype=target_dtype)
        
        # Clean up state dict to free memory
        del state_dict
        if memory_efficient:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        logging.info(f"Model loaded successfully with dtype {target_dtype}")
        return model


class ModelInitializer:
    """Automatic model initialization wrapper with bf16 safetensors support"""
    
    @staticmethod
    def auto_initialize_model(
        model_class,
        checkpoint_dir: Union[str, Path],
        prefer_safetensors: bool = True,
        target_dtype: torch.dtype = torch.bfloat16,
        device: str = "cpu",
        **kwargs
    ):
        """Automatically initialize model with best available format"""
        checkpoint_dir = Path(checkpoint_dir)
        
        try:
            format_type = SafetensorsModelLoader.detect_model_format(checkpoint_dir)
            
            # Use enhanced loader if safetensors available or prefer safetensors
            if format_type == "safetensors" or prefer_safetensors:
                logging.info("Using enhanced SafetensorsModelLoader")
                return SafetensorsModelLoader.enhanced_from_pretrained(
                    model_class, checkpoint_dir, 
                    target_dtype=target_dtype, device=device, **kwargs
                )
            else:
                # Fallback to original method
                logging.info("Using original from_pretrained method")
                return model_class.from_pretrained(checkpoint_dir)
                
        except Exception as e:
            logging.error(f"Enhanced loading failed: {e}")
            logging.info("Falling back to original from_pretrained method")
            try:
                return model_class.from_pretrained(checkpoint_dir)
            except Exception as fallback_e:
                logging.error(f"Fallback loading also failed: {fallback_e}")
                raise


def patch_wan_models():
    """Patch WanModel and VaceWanModel classes with enhanced from_pretrained"""
    
    @classmethod
    def enhanced_from_pretrained_wan(
        cls,
        checkpoint_dir: Union[str, Path],
        model_filename: Optional[str] = None,
        target_dtype: torch.dtype = torch.bfloat16,
        device: str = "cpu",
        **kwargs
    ):
        return SafetensorsModelLoader.enhanced_from_pretrained(
            cls, checkpoint_dir, model_filename, target_dtype, device, **kwargs
        )
    
    @classmethod
    def auto_from_pretrained(
        cls,
        checkpoint_dir: Union[str, Path],
        **kwargs
    ):
        return ModelInitializer.auto_initialize_model(cls, checkpoint_dir, **kwargs)
    
    # Add enhanced methods to model classes
    WanModel.from_pretrained_enhanced = enhanced_from_pretrained_wan
    VaceWanModel.from_pretrained_enhanced = enhanced_from_pretrained_wan
    WanModel.auto_from_pretrained = auto_from_pretrained
    VaceWanModel.auto_from_pretrained = auto_from_pretrained
    
    logging.info("WanModel and VaceWanModel have been enhanced with safetensors support")


# Configuration utilities
def create_bf16_safetensors_config():
    """Create a configuration for using bf16 safetensors models"""
    return {
        "use_safetensors": True,
        "target_dtype": torch.bfloat16,
        "model_filename": None,  # Auto-detect
        "device": "cpu",  # Load to CPU first for memory efficiency
        "memory_efficient": True,
        "auto_detect_dtype": True
    }


def verify_model_compatibility(checkpoint_dir: Union[str, Path]) -> Dict[str, Any]:
    """Verify model compatibility and provide recommendations"""
    checkpoint_dir = Path(checkpoint_dir)
    
    result = {
        "compatible": False,
        "format": None,
        "recommendations": [],
        "issues": []
    }
    
    try:
        format_type = SafetensorsModelLoader.detect_model_format(checkpoint_dir)
        result["format"] = format_type
        result["compatible"] = True
        
        if format_type == "safetensors":
            result["recommendations"].append("Model is in safetensors format - optimal for loading")
            if SAFETENSORS_AVAILABLE:
                result["recommendations"].append("safetensors library is available")
            else:
                result["issues"].append("safetensors library not installed")
                result["recommendations"].append("Install safetensors: pip install safetensors")
        else:
            result["recommendations"].append("Consider converting to safetensors format for better performance")
        
    except FileNotFoundError as e:
        result["issues"].append(str(e))
        result["recommendations"].append("Check checkpoint directory path")
    
    return result 
