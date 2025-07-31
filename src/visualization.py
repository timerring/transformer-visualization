"""
可视化模块
用于展示Transformer各组件的分析结果
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import torch
import numpy as np

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from .attention_analyzer import AttentionAnalyzer
from .positional_encoding import PositionalEncodingAnalyzer

class TransformerVisualizer:
    """
    Transformer可视化器
    提供各种可视化功能来展示Transformer组件的作用
    """
    def __init__(self, config):
        self.config = config
        self.attention_analyzer = AttentionAnalyzer()
        self.positional_analyzer = PositionalEncodingAnalyzer()
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

    def visualize_attention_evolution(self, attention_weights_list, save_path=None, title="Attention Evolution"):
        """
        可视化注意力权重在不同层之间的演化
        
        Args:
            attention_weights_list: 包含各层注意力权重的列表
            save_path: 保存路径
            title: 图表标题
        """
        if not attention_weights_list:
            print("警告：注意力权重列表为空")
            return
            
        # 计算平均注意力权重（跨所有头和批次）
        layer_avg_attention = []
        for layer_weights in attention_weights_list:
            # 处理不同维度的注意力权重
            if layer_weights.dim() == 4:  # [batch_size, n_heads, seq_len, seq_len]
                avg_weights = layer_weights.mean(dim=(0, 1))  # [seq_len, seq_len]
            elif layer_weights.dim() == 3:  # [n_heads, seq_len, seq_len]
                avg_weights = layer_weights.mean(dim=0)  # [seq_len, seq_len]
            elif layer_weights.dim() == 2:  # [seq_len, seq_len]
                avg_weights = layer_weights
            else:
                print(f"警告：不支持的注意力权重维度: {layer_weights.shape}")
                continue
            layer_avg_attention.append(avg_weights.detach().cpu().numpy())
        
        if not layer_avg_attention:
            print("警告：没有有效的注意力权重数据")
            return
            
        n_layers = len(layer_avg_attention)
        seq_len = layer_avg_attention[0].shape[0]
        
        # 创建子图
        fig, axes = plt.subplots(2, min(n_layers, 4), figsize=(4*min(n_layers, 4), 8))
        if n_layers == 1:
            axes = axes.reshape(2, 1)
        
        # 绘制每一层的注意力热力图
        for i in range(min(n_layers, 4)):
            if n_layers == 1:
                ax1, ax2 = axes[0, 0], axes[1, 0]
            else:
                ax1, ax2 = axes[0, i], axes[1, i]
            
            # 上层：注意力热力图
            im1 = ax1.imshow(layer_avg_attention[i], cmap='viridis', aspect='auto')
            ax1.set_title(f'The {i+1} Layer Attention Weights')
            ax1.set_xlabel('Key Position')
            ax1.set_ylabel('Query Position')
            plt.colorbar(im1, ax=ax1)
            
            # 下层：注意力分布统计
            attention_dist = layer_avg_attention[i].flatten()
            ax2.hist(attention_dist, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            ax2.set_title(f'Layer {i+1} Attention Distribution')
            ax2.set_xlabel('Attention Weight')
            ax2.set_ylabel('Frequency')
            ax2.axvline(attention_dist.mean(), color='red', linestyle='--', 
                       label=f'Mean: {attention_dist.mean():.3f}')
            ax2.legend()
        
        # 隐藏多余的子图
        for i in range(n_layers, 4):
            if n_layers == 1:
                axes[0, 0].set_visible(False)
                axes[1, 0].set_visible(False)
            else:
                axes[0, i].set_visible(False)
                axes[1, i].set_visible(False)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            # 确保目录存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        # 额外绘制层间演化图
        if n_layers > 1:
            self._plot_layer_evolution(layer_avg_attention, save_path, title)

    def _plot_layer_evolution(self, layer_avg_attention, save_path, title):
        """
        绘制层间演化图
        
        Args:
            layer_avg_attention: 各层平均注意力权重列表
            save_path: 保存路径
            title: 图表标题
        """
        n_layers = len(layer_avg_attention)
        seq_len = layer_avg_attention[0].shape[0]
        
        # 计算层间变化
        layer_changes = []
        for i in range(1, n_layers):
            change = np.abs(layer_avg_attention[i] - layer_avg_attention[i-1])
            layer_changes.append(change)
        
        # 创建演化图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 层间注意力变化热力图
        if layer_changes:
            ax1 = axes[0, 0]
            # 显示最后一层的变化
            im1 = ax1.imshow(layer_changes[-1], cmap='Reds', aspect='auto')
            ax1.set_title('Last Layer Attention Change')
            ax1.set_xlabel('Key Position')
            ax1.set_ylabel('Query Position')
            plt.colorbar(im1, ax=ax1)
        
        # 2. 对角线注意力演化
        ax2 = axes[0, 1]
        diagonal_attention = []
        for layer_attn in layer_avg_attention:
            diagonal = np.diag(layer_attn)
            diagonal_attention.append(diagonal)
        
        diagonal_attention = np.array(diagonal_attention)  # [n_layers, seq_len]
        for i in range(min(seq_len, 10)):  # 只显示前10个位置
            ax2.plot(range(1, n_layers + 1), diagonal_attention[:, i], 
                    marker='o', label=f'Position {i+1}')
        ax2.set_title('Diagonal Attention Evolution')
        ax2.set_xlabel('Layer Number')
        ax2.set_ylabel('Attention Weight')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 3. 平均注意力强度演化
        ax3 = axes[1, 0]
        mean_attention = [np.mean(layer_attn) for layer_attn in layer_avg_attention]
        ax3.plot(range(1, n_layers + 1), mean_attention, marker='o', linewidth=2, markersize=8)
        ax3.set_title('Average Attention Strength Evolution')
        ax3.set_xlabel('Layer Number')
        ax3.set_ylabel('Average Attention Weight')
        ax3.grid(True, alpha=0.3)
        
        # 4. 注意力集中度演化（熵）
        ax4 = axes[1, 1]
        attention_entropy = []
        for layer_attn in layer_avg_attention:
            # 计算每个位置的注意力熵
            entropy = -np.sum(layer_attn * np.log(layer_attn + 1e-8), axis=1)
            attention_entropy.append(np.mean(entropy))
        
        ax4.plot(range(1, n_layers + 1), attention_entropy, marker='s', linewidth=2, markersize=8, color='orange')
        ax4.set_title('Attention Concentration Evolution (Entropy)')
        ax4.set_xlabel('Layer Number')
        ax4.set_ylabel('Average Entropy')
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(f'{title} - Layer Evolution Analysis', fontsize=16)
        plt.tight_layout()
        
        # 保存演化图
        if save_path:
            base_path = os.path.splitext(save_path)[0]
            evolution_path = f"{base_path}_evolution.png"
            plt.savefig(evolution_path, dpi=300, bbox_inches='tight')
        
        plt.show()

    def visualize_model_architecture(self, save_path: str = None):
        """
        可视化Transformer模型架构
        """
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        components = {
            'Input Embedding': (1, 8),
            'Positional Encoding': (1, 7),
            'Encoder Layer 1': (2, 6),
            'Encoder Layer 2': (2, 5),
            'Encoder Layer 3': (2, 4),
            'Encoder Layer 4': (2, 3),
            'Encoder Layer 5': (2, 2),
            'Encoder Layer 6': (2, 1),
            'Decoder Layer 1': (4, 6),
            'Decoder Layer 2': (4, 5),
            'Decoder Layer 3': (4, 4),
            'Decoder Layer 4': (4, 3),
            'Decoder Layer 5': (4, 2),
            'Decoder Layer 6': (4, 1),
            'Output Layer': (5, 4)
        }
        for component, (x, y) in components.items():
            if 'Encoder' in component:
                color = 'lightblue'
            elif 'Decoder' in component:
                color = 'lightgreen'
            elif 'Input' in component or 'Positional' in component:
                color = 'lightyellow'
            else:
                color = 'lightcoral'
            ax.add_patch(plt.Rectangle((x-0.4, y-0.3), 0.8, 0.6, facecolor=color, edgecolor='black', linewidth=2))
            ax.text(x, y, component, ha='center', va='center', fontsize=8, fontweight='bold')
        for i in range(1, 6):
            ax.arrow(1.5, 6.5-i, 0.5, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')
        for i in range(1, 6):
            ax.arrow(4.5, 6.5-i, 0.5, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')
        ax.arrow(2.5, 3.5, 1, 0, head_width=0.1, head_length=0.1, fc='red', ec='red', linewidth=2)
        ax.arrow(1, 7.5, 0, -3, head_width=0.1, head_length=0.1, fc='blue', ec='blue', linewidth=2)
        ax.arrow(5, 4.5, 0, -0.5, head_width=0.1, head_length=0.1, fc='blue', ec='blue', linewidth=2)
        ax.text(1.5, 9, 'Encoder', ha='center', va='center', fontsize=14, fontweight='bold')
        ax.text(4.5, 9, 'Decoder', ha='center', va='center', fontsize=14, fontweight='bold')
        ax.text(3, 8.5, 'Cross Attention', ha='center', va='center', fontsize=12, color='red')
        ax.set_xlim(0, 6)
        ax.set_ylim(0, 10)
        ax.set_aspect('equal')
        ax.axis('off')
        plt.title('Transformer Architecture', fontsize=16, fontweight='bold', pad=20)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def visualize_component_contribution(self, ablation_results, save_path=None):
        """
        可视化各组件消融对BLEU分数的影响（条形图）
        ablation_results: dict, 形如 {component_name: bleu_diff}
        """
        plt.figure(figsize=(10, 6))
        components = list(ablation_results.keys())
        values = list(ablation_results.values())
        bars = plt.bar(components, values, color='skyblue')
        plt.xlabel('Ablation Component', fontsize=12)
        plt.ylabel('BLEU Change(%)', fontsize=12)
        plt.title('Ablation Study on BLEU Score', fontsize=14)
        plt.xticks(rotation=30)
        for bar, v in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, v, f'{v:.2f}', ha='center', va='bottom', fontsize=10)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show() 