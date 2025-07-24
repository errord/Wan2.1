# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import os
import logging
import torch
import gc
import json
from typing import Optional, Union, Dict, Any, List, Tuple
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
    def detect_model_format(checkpoint_dir: Union[str, Path]) -> Tuple[str, Dict[str, Any]]:
        """Detect the model checkpoint format in the directory
        
        Returns:
            Tuple of (format_type, format_info) where format_info contains details about the format
        """
        checkpoint_dir = Path(checkpoint_dir)
        
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_dir}")
        
        # Check for .safetensors.index.json file (multi-file with index)
        safetensors_index_files = list(checkpoint_dir.glob("*.safetensors.index.json"))
        if safetensors_index_files:
            index_file = safetensors_index_files[0]
            logging.info(f"Found safetensors index file: {index_file}")
            
            # Read index file to get file mapping
            try:
                with open(index_file, 'r') as f:
                    index_data = json.load(f)
                
                safetensors_files = []
                if 'weight_map' in index_data:
                    # Get unique safetensors files from weight_map
                    safetensors_files = list(set(index_data['weight_map'].values()))
                
                format_info = {
                    'subtype': 'indexed_multi_file',
                    'index_file': index_file,
                    'safetensors_files': safetensors_files,
                    'index_data': index_data
                }
                
                return "safetensors", format_info
                
            except Exception as e:
                logging.warning(f"Failed to read index file {index_file}: {e}")
                # Fall through to check for regular safetensors files

        # Check for safetensors files
        safetensors_files = list(checkpoint_dir.glob("*.safetensors"))
        if safetensors_files:
            if len(safetensors_files) == 1:
                logging.info(f"Found single safetensors file: {safetensors_files[0]}")
                format_info = {
                    'subtype': 'single_file',
                    'safetensors_files': safetensors_files
                }
            else:
                logging.info(f"Found {len(safetensors_files)} safetensors files (no index)")
                format_info = {
                    'subtype': 'multi_file_no_index',
                    'safetensors_files': safetensors_files
                }
            
            return "safetensors", format_info
        
        # Check for pytorch files
        pytorch_files = list(checkpoint_dir.glob("*.pth")) + list(checkpoint_dir.glob("*.bin"))
        if pytorch_files:
            logging.info(f"Found {len(pytorch_files)} pytorch files")
            format_info = {
                'subtype': 'pytorch',
                'pytorch_files': pytorch_files
            }
            return "pytorch", format_info
        
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
    def load_safetensors_indexed_state_dict(
        checkpoint_dir: Union[str, Path],
        index_file: Path,
        index_data: Dict[str, Any],
        target_dtype: torch.dtype = torch.bfloat16,
        device: str = "cpu",
        memory_efficient: bool = True
    ) -> dict:
        """Load state dict from indexed safetensors files"""
        if not SAFETENSORS_AVAILABLE:
            raise ImportError("safetensors is required for loading .safetensors files")
        
        checkpoint_dir = Path(checkpoint_dir)
        logging.info(f"Loading indexed safetensors model from {checkpoint_dir}")
        
        # Clear GPU cache before loading
        if memory_efficient and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        try:
            state_dict = {}
            weight_map = index_data.get('weight_map', {})
            
            # Group parameters by file for efficient loading
            file_to_keys = {}
            for param_name, filename in weight_map.items():
                if filename not in file_to_keys:
                    file_to_keys[filename] = []
                file_to_keys[filename].append(param_name)
            
            # Load each safetensors file
            for filename, param_names in file_to_keys.items():
                file_path = checkpoint_dir / filename
                logging.info(f"Loading {filename} with {len(param_names)} parameters")
                
                file_state_dict = safetensors.load_file(str(file_path), device=device)
                
                # Add parameters from this file to main state_dict
                for param_name in param_names:
                    if param_name in file_state_dict:
                        tensor = file_state_dict[param_name]
                        
                        # Convert to target dtype
                        if target_dtype != torch.float32 and tensor.dtype.is_floating_point:
                            tensor = tensor.to(dtype=target_dtype)
                        
                        state_dict[param_name] = tensor
                
                # Clean up file state dict
                del file_state_dict
                if memory_efficient:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            logging.info(f"Successfully loaded indexed safetensors model with {len(state_dict)} parameters")
            return state_dict
            
        except Exception as e:
            logging.error(f"Error loading indexed safetensors from {checkpoint_dir}: {e}")
            raise
    
    @staticmethod
    def load_safetensors_multi_file_state_dict(
        safetensors_files: List[Path],
        target_dtype: torch.dtype = torch.bfloat16,
        device: str = "cpu",
        memory_efficient: bool = True
    ) -> dict:
        """Load state dict from multiple safetensors files (no index)"""
        if not SAFETENSORS_AVAILABLE:
            raise ImportError("safetensors is required for loading .safetensors files")
        
        logging.info(f"Loading {len(safetensors_files)} safetensors files without index")
        
        # Clear GPU cache before loading
        if memory_efficient and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        try:
            state_dict = {}
            
            # Load each safetensors file
            for file_path in safetensors_files:
                logging.info(f"Loading {file_path.name}")
                
                file_state_dict = safetensors.load_file(str(file_path), device=device)
                
                # Add parameters from this file to main state_dict
                for param_name, tensor in file_state_dict.items():
                    # Convert to target dtype
                    if target_dtype != torch.float32 and tensor.dtype.is_floating_point:
                        tensor = tensor.to(dtype=target_dtype)
                    
                    if param_name in state_dict:
                        logging.warning(f"Duplicate parameter {param_name} found in {file_path.name}")
                    
                    state_dict[param_name] = tensor
                
                # Clean up file state dict
                del file_state_dict
                if memory_efficient:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            logging.info(f"Successfully loaded multi-file safetensors model with {len(state_dict)} parameters")
            return state_dict
            
        except Exception as e:
            logging.error(f"Error loading multi-file safetensors: {e}")
            raise
    
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
        force_loading_method: Optional[str] = None,  # 'single_file', 'multi_file_no_index', 'indexed_multi_file', 'pytorch'
        safetensors_files: Optional[List[str]] = None,  # User-specified list of safetensors files
        index_file: Optional[str] = None,  # User-specified index file path
        **kwargs
    ):
        """Enhanced from_pretrained method supporting safetensors and dtype conversion
        
        Args:
            model_class: The model class to instantiate
            checkpoint_dir: Directory containing model files
            model_filename: Specific model file name (for single file loading)
            target_dtype: Target dtype for model weights
            device: Device to load model to
            memory_efficient: Enable memory efficient loading
            auto_detect_dtype: Auto-detect and log dtype information
            force_loading_method: Force specific loading method ('single_file', 'multi_file_no_index', 'indexed_multi_file', 'pytorch')
            safetensors_files: User-specified list of safetensors files (relative to checkpoint_dir)
            index_file: User-specified index file path (relative to checkpoint_dir)
            **kwargs: Additional arguments for model class
        """
        checkpoint_dir = Path(checkpoint_dir)
        
        # Handle user-specified loading method
        if force_loading_method is not None:
            logging.info(f"User forced loading method: {force_loading_method}")
            
            if force_loading_method == "indexed_multi_file":
                if index_file is None:
                    # Auto-find index file
                    index_files = list(checkpoint_dir.glob("*.safetensors.index.json"))
                    if not index_files:
                        raise FileNotFoundError("No index file found for indexed_multi_file method")
                    index_file = index_files[0].name
                
                index_path = checkpoint_dir / index_file
                with open(index_path, 'r') as f:
                    index_data = json.load(f)
                
                safetensors_file_list = []
                if safetensors_files is not None:
                    safetensors_file_list = [checkpoint_dir / f for f in safetensors_files]
                else:
                    if 'weight_map' in index_data:
                        safetensors_file_list = [checkpoint_dir / f for f in set(index_data['weight_map'].values())]
                
                format_info = {
                    'subtype': 'indexed_multi_file',
                    'index_file': index_path,
                    'safetensors_files': safetensors_file_list,
                    'index_data': index_data
                }
                format_type = "safetensors"
                
            elif force_loading_method == "multi_file_no_index":
                if safetensors_files is not None:
                    safetensors_file_list = [checkpoint_dir / f for f in safetensors_files]
                else:
                    safetensors_file_list = list(checkpoint_dir.glob("*.safetensors"))
                    if not safetensors_file_list:
                        raise FileNotFoundError("No safetensors files found for multi_file_no_index method")
                
                format_info = {
                    'subtype': 'multi_file_no_index',
                    'safetensors_files': safetensors_file_list
                }
                format_type = "safetensors"
                
            elif force_loading_method == "single_file":
                if model_filename is None:
                    if safetensors_files is not None and len(safetensors_files) == 1:
                        model_filename = safetensors_files[0]
                    else:
                        # Auto-find single safetensors file
                        safetensors_file_list = list(checkpoint_dir.glob("*.safetensors"))
                        if not safetensors_file_list:
                            raise FileNotFoundError("No safetensors files found for single_file method")
                        model_filename = safetensors_file_list[0].name
                
                format_info = {
                    'subtype': 'single_file',
                    'safetensors_files': [checkpoint_dir / model_filename]
                }
                format_type = "safetensors"
                
            elif force_loading_method == "pytorch":
                if model_filename is None:
                    pytorch_files = list(checkpoint_dir.glob("*.pth")) + list(checkpoint_dir.glob("*.bin"))
                    if not pytorch_files:
                        raise FileNotFoundError("No pytorch files found for pytorch method")
                    model_filename = pytorch_files[0].name
                
                format_info = {
                    'subtype': 'pytorch',
                    'pytorch_files': [checkpoint_dir / model_filename]
                }
                format_type = "pytorch"
            else:
                raise ValueError(f"Invalid force_loading_method: {force_loading_method}")
        else:
            # Detect model format automatically
            format_type, format_info = cls.detect_model_format(checkpoint_dir)
        
        if model_filename is None and force_loading_method is None:
            if format_type == "safetensors":
                if format_info['subtype'] == 'indexed_multi_file':
                    # For indexed multi-file, we don't need a single filename
                    # We'll use the index to load all files
                    model_filename = None  # Will be handled specially
                else:
                    # Look for safetensors files (already Path objects from glob)
                    safetensors_files = format_info['safetensors_files']
                    # Prefer model files with 'model' in the name
                    model_files = [f for f in safetensors_files if 'model' in f.name.lower()]
                    if model_files:
                        model_filename = model_files[0].name
                    else:
                        model_filename = safetensors_files[0].name
            else:
                # Look for pytorch files (already Path objects from glob)
                pytorch_files = format_info['pytorch_files']
                # Prefer model files with 'model' in the name
                model_files = [f for f in pytorch_files if 'model' in f.name.lower()]
                if model_files:
                    model_filename = model_files[0].name
                else:
                    model_filename = pytorch_files[0].name
        
        # Set checkpoint_path based on the loading strategy
        if format_type == "safetensors" and format_info['subtype'] == 'indexed_multi_file':
            checkpoint_path = None  # Will load from index
        else:
            checkpoint_path = checkpoint_dir / model_filename
        
        # Get model info for logging (skip for indexed multi-file)
        model_info = {}
        if checkpoint_path is not None:
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
        
        # Determine loading method based on format and subtype
        if format_type == "safetensors":
            if format_info['subtype'] == 'indexed_multi_file':
                # Load from index
                state_dict = cls.load_safetensors_indexed_state_dict(
                    checkpoint_dir, format_info['index_file'], format_info['index_data'], 
                    target_dtype, device, memory_efficient
                )
            elif format_info['subtype'] == 'multi_file_no_index':
                # Load multiple files without index (already Path objects from glob)
                state_dict = cls.load_safetensors_multi_file_state_dict(
                    format_info['safetensors_files'], target_dtype, device, memory_efficient
                )
            else:  # single_file
                # Load single file
                state_dict = cls.load_safetensors_state_dict(
                    checkpoint_path, target_dtype, device, memory_efficient
                )
        else:  # pytorch format
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
            format_type, format_info = SafetensorsModelLoader.detect_model_format(checkpoint_dir)
            
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
        "subtype": None,
        "recommendations": [],
        "issues": []
    }
    
    try:
        format_type, format_info = SafetensorsModelLoader.detect_model_format(checkpoint_dir)
        result["format"] = format_type
        result["subtype"] = format_info.get("subtype", "unknown")
        result["compatible"] = True
        
        if format_type == "safetensors":
            subtype = format_info["subtype"]
            if subtype == "indexed_multi_file":
                result["recommendations"].append("Model uses indexed multi-file safetensors format - excellent for large models")
            elif subtype == "multi_file_no_index":
                result["recommendations"].append("Model uses multi-file safetensors without index - will load all files")
            else:  # single_file
                result["recommendations"].append("Model uses single-file safetensors format - optimal for loading")
            
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
