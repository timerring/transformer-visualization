# Transformer Visualization and Analysis

## 项目概述

本项目旨在通过机器翻译任务，深入展示Transformer架构中各个核心组件的作用和贡献。通过可视化和实验分析，来间接理解Transformer原理。

## 实验目标

1. **展示Transformer各组件作用**：
   - 多头自注意力机制
   - 位置编码
   - 前馈神经网络
   - 残差连接和层归一化

2. **消融实验**：
   - 移除不同组件的影响
   - 分析各组件对性能的贡献

3. **注意力可视化**：
   - 展示注意力权重分布
   - 分析不同头的注意力模式

4. **性能对比**：
   - 与RNN、LSTM等传统模型对比
   - 不同配置下的性能分析

## 环境要求

- Python 3.8+
- PyTorch 1.9+
- NumPy
- Matplotlib
- Seaborn
- tqdm

## 快速开始

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 组件分析：
```bash
python experiments/component_analysis.py
```

3. 消融实验：

```bash
python experiments/ablation_study.py
```

4. 注意力可视化：
```bash
python experiments/attention_visualization.py
```