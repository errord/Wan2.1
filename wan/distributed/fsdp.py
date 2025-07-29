# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
from functools import partial

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
from torch.distributed.utils import _free_storage


def create_block_group_policy(model, num_groups=4):
    """
    Create auto wrap policy that groups consecutive blocks together.
    
    Args:
        model: The model containing blocks
        num_groups: Number of groups to split blocks into (default: 4)
    
    Returns:
        Function that returns True for modules that should be wrapped
    """
    total_blocks = len(model.blocks)
    blocks_per_group = total_blocks // num_groups
    
    # Create groups of consecutive block indices
    block_groups = []
    for i in range(num_groups):
        start_idx = i * blocks_per_group
        if i == num_groups - 1:  # Last group gets remaining blocks
            end_idx = total_blocks
        else:
            end_idx = (i + 1) * blocks_per_group
        
        group_blocks = set(model.blocks[start_idx:end_idx])
        block_groups.append(group_blocks)
    
    def should_wrap(module):
        # Check if this module belongs to any block group
        for group in block_groups:
            if module in group:
                return True
        return False
    
    return should_wrap


def shard_model(
    model,
    device_id,
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
    buffer_dtype=torch.float32,
    process_group=None,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    sync_module_states=True,
    num_block_groups=4,
):
    """
    Shard model using FSDP with block grouping strategy.
    
    Args:
        num_block_groups: Number of groups to split transformer blocks into (default: 4)
    """
    block_group_policy = create_block_group_policy(model, num_block_groups)
    
    model = FSDP(
        module=model,
        process_group=process_group,
        sharding_strategy=sharding_strategy,
        auto_wrap_policy=partial(
            lambda_auto_wrap_policy, lambda_fn=block_group_policy),
        mixed_precision=MixedPrecision(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            buffer_dtype=buffer_dtype),
        device_id=device_id,
        sync_module_states=sync_module_states)
    return model


def shard_model_cpu_init(
    model,
    device_id,
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
    buffer_dtype=torch.float32,
    process_group=None,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    sync_module_states=False,  # Key fix: disable sync_module_states for CPU init
    num_block_groups=4,
):
    """
    Initialize FSDP with CPU-first approach to reduce GPU memory peak.
    
    Args:
        num_block_groups: Number of groups to split transformer blocks into (default: 4)
    """
    # Ensure model is on CPU and clear GPU cache
    model = model.cpu()
    torch.cuda.empty_cache()
    
    block_group_policy = create_block_group_policy(model, num_block_groups)
    
    # Initialize FSDP with sync_module_states=False to allow CPU initialization
    model = FSDP(
        module=model,
        process_group=process_group,
        sharding_strategy=sharding_strategy,
        auto_wrap_policy=partial(
            lambda_auto_wrap_policy, lambda_fn=block_group_policy),
        mixed_precision=MixedPrecision(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            buffer_dtype=buffer_dtype),
        device_id=None,  # Keep on CPU during initialization
        sync_module_states=sync_module_states)  # Disabled for CPU init
    
    # Move to GPU after FSDP wrapping
    model = model.to(f"cuda:{device_id}")
    
    # Manual synchronization across ranks if needed
    if sync_module_states and torch.distributed.is_initialized():
        torch.distributed.barrier()
    
    return model


def shard_model_with_cpu_offload(
    model,
    device_id,
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
    buffer_dtype=torch.float32,
    process_group=None,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    sync_module_states=True,
    num_block_groups=4,
):
    """
    Initialize FSDP with CPU offload to minimize GPU memory usage.
    This keeps parameters on CPU and only loads them to GPU when needed.
    
    Args:
        num_block_groups: Number of groups to split transformer blocks into (default: 4)
    """
    from torch.distributed.fsdp import CPUOffload
    
    # Ensure model is on CPU and clear GPU cache aggressively
    model = model.cpu()
    torch.cuda.empty_cache()
    
    block_group_policy = create_block_group_policy(model, num_block_groups)
    
    # Use CPU offload to minimize GPU memory usage
    model = FSDP(
        module=model,
        process_group=process_group,
        sharding_strategy=sharding_strategy,
        auto_wrap_policy=partial(
            lambda_auto_wrap_policy, lambda_fn=block_group_policy),
        mixed_precision=MixedPrecision(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            buffer_dtype=buffer_dtype),
        cpu_offload=CPUOffload(offload_params=True),  # Key: keep params on CPU
        device_id=device_id,
        sync_module_states=sync_module_states)
    
    return model

def free_model(model):
    for m in model.modules():
        if isinstance(m, FSDP):
            _free_storage(m._handle.flat_param.data)
    del model
    gc.collect()
    torch.cuda.empty_cache()