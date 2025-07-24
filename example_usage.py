#!/usr/bin/env python3
"""
Example usage of the enhanced model loader with safetensors support.
Shows how to use the new user-specified parameters.
"""

# Example usage of the enhanced model loader

def example_auto_detection():
    """Example 1: Automatic detection (original behavior)"""
    print("=== Example 1: Automatic Detection ===")
    
    # Original usage - automatic detection
    """
    from wan.utils.model_loader import SafetensorsModelLoader
    from wan.modules.model import WanModel
    
    model = SafetensorsModelLoader.enhanced_from_pretrained(
        WanModel,
        checkpoint_dir="path/to/model/dir",
        target_dtype=torch.bfloat16,
        device="cpu"
    )
    """
    print("✓ Automatically detects and loads the best available format")

def example_force_single_file():
    """Example 2: Force single file loading"""
    print("\n=== Example 2: Force Single File Loading ===")
    
    # Force single file loading
    """
    model = SafetensorsModelLoader.enhanced_from_pretrained(
        WanModel,
        checkpoint_dir="path/to/model/dir",
        force_loading_method="single_file",
        model_filename="model.safetensors",  # Optional: specify exact file
        target_dtype=torch.bfloat16
    )
    """
    print("✓ Forces loading from a single .safetensors file")

def example_force_multi_file_no_index():
    """Example 3: Force multi-file loading without index"""
    print("\n=== Example 3: Force Multi-file (No Index) Loading ===")
    
    # Force multi-file loading without index
    """
    model = SafetensorsModelLoader.enhanced_from_pretrained(
        WanModel,
        checkpoint_dir="path/to/model/dir",
        force_loading_method="multi_file_no_index",
        safetensors_files=[  # Optional: specify exact files
            "model-00001-of-00003.safetensors",
            "model-00002-of-00003.safetensors", 
            "model-00003-of-00003.safetensors"
        ],
        target_dtype=torch.bfloat16
    )
    """
    print("✓ Forces loading from multiple .safetensors files without index")

def example_force_indexed_multi_file():
    """Example 4: Force indexed multi-file loading"""
    print("\n=== Example 4: Force Indexed Multi-file Loading ===")
    
    # Force indexed multi-file loading
    """
    model = SafetensorsModelLoader.enhanced_from_pretrained(
        WanModel,
        checkpoint_dir="path/to/model/dir",
        force_loading_method="indexed_multi_file",
        index_file="model.safetensors.index.json",  # Optional: specify index file
        safetensors_files=[  # Optional: specify subset of files to load
            "model-00001-of-00003.safetensors",
            "model-00002-of-00003.safetensors"
        ],
        target_dtype=torch.bfloat16
    )
    """
    print("✓ Forces loading using safetensors index file")

def example_force_pytorch():
    """Example 5: Force PyTorch loading"""
    print("\n=== Example 5: Force PyTorch Loading ===")
    
    # Force PyTorch loading
    """
    model = SafetensorsModelLoader.enhanced_from_pretrained(
        WanModel,
        checkpoint_dir="path/to/model/dir",
        force_loading_method="pytorch",
        model_filename="model.pth",  # Optional: specify exact file
        target_dtype=torch.bfloat16
    )
    """
    print("✓ Forces loading from PyTorch .pth/.bin files")

def example_with_enhanced_methods():
    """Example 6: Using enhanced methods on model classes"""
    print("\n=== Example 6: Using Enhanced Methods ===")
    
    # After calling patch_wan_models(), you can use enhanced methods
    """
    from wan.utils.model_loader import patch_wan_models
    from wan.modules.model import WanModel
    
    # Patch the model classes with enhanced methods
    patch_wan_models()
    
    # Now you can use enhanced methods directly on model classes
    model = WanModel.from_pretrained_enhanced(
        checkpoint_dir="path/to/model/dir",
        force_loading_method="indexed_multi_file",
        target_dtype=torch.bfloat16
    )
    
    # Or use auto initialization
    model = WanModel.auto_from_pretrained(
        checkpoint_dir="path/to/model/dir"
    )
    """
    print("✓ Enhanced methods available on WanModel and VaceWanModel classes")

def example_model_compatibility_check():
    """Example 7: Check model compatibility"""
    print("\n=== Example 7: Model Compatibility Check ===")
    
    # Check model compatibility before loading
    """
    from wan.utils.model_loader import verify_model_compatibility
    
    compatibility = verify_model_compatibility("path/to/model/dir")
    print(f"Format: {compatibility['format']}")
    print(f"Subtype: {compatibility['subtype']}")
    print(f"Compatible: {compatibility['compatible']}")
    print(f"Recommendations: {compatibility['recommendations']}")
    print(f"Issues: {compatibility['issues']}")
    """
    print("✓ Check compatibility and get recommendations before loading")

def main():
    """Run all examples"""
    print("Enhanced Model Loader Usage Examples")
    print("=" * 50)
    
    example_auto_detection()
    example_force_single_file()
    example_force_multi_file_no_index()
    example_force_indexed_multi_file()
    example_force_pytorch()
    example_with_enhanced_methods()
    example_model_compatibility_check()
    
    print("\n" + "=" * 50)
    print("Key Benefits:")
    print("✓ Automatic detection with proper priority (index > multi > single)")
    print("✓ User control over loading method")
    print("✓ Support for bf16 quantization")
    print("✓ Memory-efficient loading")
    print("✓ Compatibility checking")
    print("✓ Enhanced error handling")

if __name__ == "__main__":
    main() 