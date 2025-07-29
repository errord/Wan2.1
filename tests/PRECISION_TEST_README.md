# Wan2.1 精度性能测试套件

本测试套件专门用于测试和分析fp32、fp16、bf16三种精度类型在Wan2.1模型上的性能表现，并评估分布式环境下网络通讯对精度优化收益的影响。

## 核心测试内容

### 1. 单GPU精度性能测试
- **目标**: 对比fp32、fp16、bf16三种精度的计算性能
- **测试模型**: 针对RTX 5090 32GB显存优化的大型Transformer模型
- **模型规模**: 自动计算最优参数量，确保充分利用显存而不溢出
- **测试指标**: 
  - 前向传播耗时
  - 反向传播耗时
  - 总训练时间
  - 显存使用量
  - 吞吐量(tokens/s)

### 2. 分布式精度性能测试
- **目标**: 验证网络通讯是否会消耗掉fp32→fp16/bf16的计算提升
- **测试场景**: FSDP模式下的多GPU分布式训练
- **关键分析**: 
  - 纯计算时间 vs 总时间(包含通讯)
  - 网络通讯开销占比
  - 精度优化收益在分布式环境下的保留程度

## 文件结构

```
tests/
├── precision_performance_test.py    # 主测试脚本(综合测试)
├── quick_precision_test.py          # 快速验证脚本
├── run_precision_test.sh            # 便捷运行脚本
├── analyze_precision_results.py     # 结果分析工具
└── PRECISION_TEST_README.md         # 本文档
```

## 快速开始

### 1. 快速验证测试

```bash
cd tests/
python quick_precision_test.py
```

约30秒内完成基础功能验证，输出三种精度的性能对比。

### 2. 单GPU完整测试

```bash
# 只测试单GPU性能
python precision_performance_test.py --test_type single --save_results

# 或使用shell脚本
./run_precision_test.sh single 1 20
```

### 3. 分布式完整测试

```bash
# 测试2卡分布式性能
python precision_performance_test.py --test_type distributed --num_gpus 2 --save_results

# 或使用shell脚本
./run_precision_test.sh distributed 2 20
```

### 4. 完整测试套件

```bash
# 运行所有测试
./run_precision_test.sh all 2 20
```

## 命令行参数说明

### precision_performance_test.py

```bash
python precision_performance_test.py [选项]

主要选项:
  --test_type {single,distributed,all}  测试类型 (默认: all)
  --gpu_id GPU_ID                      单GPU测试的GPU编号 (默认: 0)
  --num_gpus NUM_GPUS                  分布式测试的GPU数量 (默认: 2)
  --batch_sizes [SIZE ...]             要测试的批次大小 (默认: [4, 8, 16])
  --seq_lengths [LEN ...]              要测试的序列长度 (默认: [1024, 2048])
  --num_iterations NUM                 每个测试的迭代次数 (默认: 20)
  --save_results                       保存结果到JSON文件
```

### 运行脚本参数

```bash
./run_precision_test.sh [test_type] [num_gpus] [iterations]

参数:
  test_type    : single | distributed | all (默认: all)
  num_gpus     : GPU数量 (默认: 2)
  iterations   : 迭代次数 (默认: 20)
```

## 模型配置策略

### 针对RTX 5090 32GB的优化

测试脚本会根据不同精度自动计算最优模型配置:

- **目标显存使用**: 28GB (留4GB给激活值和开销)
- **基础配置**: hidden_size=8192, num_layers=24
- **自适应缩放**: 根据精度类型调整模型大小
- **内存安全**: 动态检测并避免OOM错误

### 模型架构

```python
# 基础Transformer架构
- Embedding层: vocab_size=32000
- Transformer块: Multi-head Attention + MLP
- 注意力头数: hidden_size // 128 
- MLP中间层: hidden_size * 4
- 输出层: 与词汇表大小匹配
```

## 结果分析

### 自动分析

运行测试后，脚本会自动输出:

1. **单GPU性能对比表**
   - 各精度的前向/反向/总耗时
   - 显存使用量对比
   - 吞吐量对比

2. **分布式性能分析表**
   - 计算时间 vs 通讯时间
   - 网络开销占比
   - 精度收益保留率

3. **优化建议**
   - 基于测试结果的具体建议
   - 精度选择指导
   - 分布式优化建议

### 手动深度分析

```bash
# 生成详细分析报告
python analyze_precision_results.py --results_dir results/

# 查看生成的报告
cat precision_performance_report.txt
```

## 预期性能基准

### 典型硬件性能参考

| 硬件类型 | FP16加速比 | BF16加速比 | 显存节省 | 分布式效率保留 |
|---------|-----------|-----------|----------|---------------|
| RTX 5090 | 1.5-2.0x  | 1.3-1.7x  | 40-50%   | 80-90%       |
| RTX 4090 | 1.4-1.8x  | 1.2-1.5x  | 35-45%   | 75-85%       |
| A100     | 1.6-2.2x  | 1.4-1.8x  | 45-55%   | 85-95%       |

### 关键性能指标

1. **单GPU加速比**
   - FP16优秀: >1.5x
   - FP16良好: >1.2x
   - BF16优秀: >1.3x
   - BF16良好: >1.1x

2. **分布式效率保留**
   - 优秀: >90% (网络开销很小)
   - 良好: >80% (网络开销适中)
   - 需要优化: <80% (网络严重影响精度收益)

## 关键测试结论分析

### 1. 网络通讯影响评估

测试会回答核心问题：**网络通讯是否会消耗掉fp32→fp16(bf16)的计算提升？**

**评估维度**:
- **计算加速比**: 纯计算时间的提升倍数
- **实际加速比**: 包含通讯后的总体提升倍数
- **收益保留率**: 实际加速比/计算加速比 × 100%

**典型结论**:
- 收益保留率 >90%: 网络影响很小，精度优化收益基本保留
- 收益保留率 80-90%: 网络有一定影响，但精度优化仍有价值
- 收益保留率 <80%: 网络严重影响精度收益，需要优化网络或调整策略

### 2. 精度选择指导

基于测试结果，脚本会给出精度选择建议:

- **推荐FP16**: 计算加速比>1.5x 且 分布式效率保留>80%
- **推荐BF16**: 计算加速比>1.3x 且 需要更好的数值稳定性
- **混合精度**: 单精度收益有限但仍有价值的情况
- **保持FP32**: 精度收益很小或分布式效率损失严重

## 故障排除

### 常见问题

1. **CUDA内存不足**
   ```bash
   # 减少批次大小和序列长度
   python precision_performance_test.py --batch_sizes 2 4 --seq_lengths 512 1024
   ```

2. **模型过大导致OOM**
   - 脚本会自动调整模型大小
   - 如仍有问题，可手动修改target_memory_gb参数

3. **分布式测试失败**
   ```bash
   # 确保多GPU可用
   nvidia-smi
   
   # 尝试不同端口
   python precision_performance_test.py --master_port 12357
   ```

4. **性能结果异常**
   - 确保GPU没有被其他程序占用
   - 检查GPU散热和电源供应
   - 尝试增加预热轮数

### 性能优化建议

1. **硬件层面**
   - 确保PCIe 4.0/5.0 x16带宽
   - 使用高速NVLink(如果可用)
   - 确保充足的电源和散热

2. **软件层面**
   - 使用最新的PyTorch和CUDA版本
   - 启用Tensor Core优化
   - 合理设置环境变量

## 扩展测试

### 自定义模型配置

可以修改源码中的模型配置:

```python
# 在 precision_performance_test.py 中修改
def _calculate_model_configs(self):
    base_hidden = 8192      # 调整基础隐藏层大小
    base_layers = 24        # 调整基础层数
    self.target_memory_gb = 28  # 调整目标显存使用量
```

### 多节点测试

```bash
# 主节点 (机器1)
python precision_performance_test.py \
    --test_type distributed \
    --master_addr 192.168.1.100 \
    --master_port 12355

# 工作节点 (机器2, 3, ...)
# 需要正确设置RANK和WORLD_SIZE环境变量
```

## 结果解读示例

### 单GPU测试结果示例

```
Performance relative to FP32:
Precision  Speedup    Memory     Throughput   Efficiency
Type       Gain       Reduction  Gain         Score     
----------------------------------------------------------
FP32       1.00x      1.00x      1.00x        1.00
FP16       1.68x      1.87x      1.68x        1.77
BF16       1.52x      1.84x      1.52x        1.67
```

**解读**: FP16提供了68%的加速，同时减少了47%的显存使用，效率得分1.77表现优秀。

### 分布式测试结果示例

```
Communication Impact on Precision Benefits:
Precision  Compute    Total      Benefit    Network
Type       Speedup    Speedup    Retention  Impact
----------------------------------------------------
FP32       1.00x      1.00x      100.0      0.0
FP16       1.65x      1.42x      86.1       13.9
BF16       1.48x      1.31x      88.5       11.5
```

**解读**: FP16的计算加速比为1.65x，但由于网络开销，实际加速比降至1.42x，收益保留率86.1%，属于良好水平。

## 最佳实践建议

1. **测试频率**: 每次重大模型架构变更后都应重新测试
2. **环境控制**: 测试期间确保GPU独占，避免其他负载
3. **多次测试**: 运行多次取平均值，确保结果稳定性
4. **结果归档**: 保存测试结果用于后续对比和分析
5. **渐进优化**: 从单GPU开始，逐步扩展到分布式测试

这个测试套件为Wan2.1模型的精度优化提供了科学的测试方法和数据支撑，帮助您做出最佳的精度选择和分布式配置决策。