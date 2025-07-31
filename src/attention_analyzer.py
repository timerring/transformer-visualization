"""
注意力分析器模块
用于分析和可视化注意力权重
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from typing import Optional
import networkx as nx

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class AttentionAnalyzer:
    """
    注意力分析器
    用于分析和可视化注意力权重
    """
    
    def __init__(self):
        """初始化注意力分析器"""
        pass
    
    def visualize_attention_weights(self, attention_weights, save_path=None, title="Attention Weights"):
        # 兼容不同维度
        if attention_weights.dim() == 4:
            batch_size, n_heads, seq_len, _ = attention_weights.size()
        elif attention_weights.dim() == 3:
            n_heads, seq_len, _ = attention_weights.size()
            batch_size = 1
            attention_weights = attention_weights.unsqueeze(0)  # [1, n_heads, seq_len, seq_len]
        elif attention_weights.dim() == 2:
            seq_len, _ = attention_weights.size()
            batch_size = 1
            n_heads = 1
            attention_weights = attention_weights.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        else:
            raise ValueError(f"Unsupported attention_weights shape: {attention_weights.shape}")
        
        # 选择第一个样本进行可视化
        attention = attention_weights[0]  # [n_heads, seq_len, seq_len]
        
        # 创建子图
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for i in range(min(n_heads, 8)):
            ax = axes[i]
            im = ax.imshow(attention[i], cmap='viridis', aspect='auto')
            ax.set_title(f'Attention Head {i+1}')
            
            # 添加颜色条
            plt.colorbar(im, ax=ax)
        
        # 隐藏多余的子图
        for i in range(n_heads, 8):
            axes[i].set_visible(False)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def analyze_attention_patterns(self, attention_weights: torch.Tensor) -> dict:
        """
        分析注意力模式
        
        Args:
            attention_weights: 注意力权重张量 [batch_size, n_heads, seq_len, seq_len]
            
        Returns:
            分析结果字典
        """
        batch_size, n_heads, seq_len, _ = attention_weights.size()
        
        # 计算每个头的平均注意力分布
        mean_attention = attention_weights.mean(dim=0)  # [n_heads, seq_len, seq_len]
        
        # 计算注意力熵（衡量注意力的集中程度）
        attention_entropy = -torch.sum(attention_weights * torch.log(attention_weights + 1e-8), dim=-1)
        mean_entropy = attention_entropy.mean(dim=0)  # [n_heads, seq_len]
        
        # 计算对角线注意力（自注意力）
        diagonal_attention = torch.diagonal(attention_weights, dim1=-2, dim2=-1)  # [batch_size, n_heads, seq_len]
        mean_diagonal = diagonal_attention.mean(dim=0)  # [n_heads, seq_len]
        
        # 计算局部注意力（相邻位置的注意力）
        local_attention = []
        for i in range(seq_len - 1):
            local_attn = attention_weights[:, :, i, i:i+2].mean(dim=-1)  # [batch_size, n_heads]
            local_attention.append(local_attn)
        local_attention = torch.stack(local_attention, dim=1)  # [batch_size, n_heads, seq_len-1]
        mean_local = local_attention.mean(dim=0)  # [n_heads, seq_len-1]
        
        return {
            'mean_attention': mean_attention.detach().cpu().numpy(),
            'attention_entropy': mean_entropy.detach().cpu().numpy(),
            'diagonal_attention': mean_diagonal.detach().cpu().numpy(),
            'local_attention': mean_local.detach().cpu().numpy(),
            'attention_weights': attention_weights.detach().cpu().numpy()
        }
    
    def plot_attention_statistics(self, analysis_results: dict, save_path: str = None):
        """
        绘制注意力统计图
        
        Args:
            analysis_results: 分析结果字典
            save_path: 保存路径
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 注意力熵分布
        ax1 = axes[0, 0]
        entropy_data = analysis_results['attention_entropy']
        for i in range(entropy_data.shape[0]):
            ax1.plot(entropy_data[i], label=f'Head {i+1}')
        ax1.set_title('Attention Entropy Distribution')
        ax1.set_xlabel('Sequence Position')
        ax1.set_ylabel('Entropy Value')
        ax1.legend()
        
        # 对角线注意力
        ax2 = axes[0, 1]
        diagonal_data = analysis_results['diagonal_attention']
        for i in range(diagonal_data.shape[0]):
            ax2.plot(diagonal_data[i], label=f'Head {i+1}')
        ax2.set_title('Diagonal Attention (Self-Attention)')
        ax2.set_xlabel('Sequence Position')
        ax2.set_ylabel('Attention Weight')
        ax2.legend()
        
        # 局部注意力
        ax3 = axes[1, 0]
        local_data = analysis_results['local_attention']
        for i in range(local_data.shape[0]):
            ax3.plot(local_data[i], label=f'Head {i+1}')
        ax3.set_title('Local Attention (Adjacent Positions)')
        ax3.set_xlabel('Sequence Position')
        ax3.set_ylabel('Attention Weight')
        ax3.legend()
        
        # 平均注意力热力图
        ax4 = axes[1, 1]
        mean_attn = analysis_results['mean_attention'].mean(axis=0)
        im = ax4.imshow(mean_attn, cmap='viridis', aspect='auto')
        ax4.set_title('Average Attention Weights')
        ax4.set_xlabel('Key Position')
        ax4.set_ylabel('Query Position')
        plt.colorbar(im, ax=ax4)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def analyze_attention_head_specialization(self, attention_weights: torch.Tensor) -> dict:
        """
        分析注意力头的专业化程度
        
        Args:
            attention_weights: 注意力权重张量 [batch_size, n_heads, seq_len, seq_len]
            
        Returns:
            分析结果字典
        """
        batch_size, n_heads, seq_len, _ = attention_weights.size()
        
        # 计算每个头的注意力集中度
        attention_entropy = -torch.sum(attention_weights * torch.log(attention_weights + 1e-8), dim=-1)
        mean_entropy_per_head = attention_entropy.mean(dim=(0, 2))  # [n_heads]
        
        # 计算每个头的最大注意力权重
        max_attention_per_head = attention_weights.max(dim=-1)[0].mean(dim=(0, 1))  # [n_heads]
        
        # 计算每个头的注意力分布方差
        attention_variance = torch.var(attention_weights, dim=-1).mean(dim=(0, 1))  # [n_heads]
        
        # 计算每个头关注的位置范围
        position_range = []
        for head in range(n_heads):
            head_attention = attention_weights[:, head, :, :]  # [batch_size, seq_len, seq_len]
            # 找到每个位置的最大注意力位置
            max_positions = head_attention.max(dim=-1)[1]  # [batch_size, seq_len]
            # 计算位置范围
            ranges = max_positions.max(dim=1)[0] - max_positions.min(dim=1)[0]  # [batch_size]
            position_range.append(ranges.mean().item())
        
        return {
            'entropy_per_head': mean_entropy_per_head.detach().cpu().numpy(),
            'max_attention_per_head': max_attention_per_head.detach().cpu().numpy(),
            'variance_per_head': attention_variance.detach().cpu().numpy(),
            'position_range_per_head': position_range
        }
    
    def plot_head_specialization(self, specialization_results: dict, save_path: str = None):
        """
        绘制注意力头专业化分析图
        
        Args:
            specialization_results: 专业化分析结果
            save_path: 保存路径
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        n_heads = len(specialization_results['entropy_per_head'])
        head_indices = range(1, n_heads + 1)
        
        # 注意力熵
        ax1 = axes[0, 0]
        ax1.bar(head_indices, specialization_results['entropy_per_head'])
        ax1.set_title('Entropy Values for Each Attention Head')
        ax1.set_xlabel('Attention Head')
        ax1.set_ylabel('Entropy Value')
        ax1.set_xticks(head_indices)
        
        # 最大注意力权重
        ax2 = axes[0, 1]
        ax2.bar(head_indices, specialization_results['max_attention_per_head'])
        ax2.set_title('Maximum Weights for Each Attention Head')
        ax2.set_xlabel('Attention Head')
        ax2.set_ylabel('Maximum Weight')
        ax2.set_xticks(head_indices)
        
        # 注意力方差
        ax3 = axes[1, 0]
        ax3.bar(head_indices, specialization_results['variance_per_head'])
        ax3.set_title('Variance for Each Attention Head')
        ax3.set_xlabel('Attention Head')
        ax3.set_ylabel('Variance')
        ax3.set_xticks(head_indices)
        
        # 位置范围
        ax4 = axes[1, 1]
        ax4.bar(head_indices, specialization_results['position_range_per_head'])
        ax4.set_title('Position Range for Each Attention Head')
        ax4.set_xlabel('Attention Head')
        ax4.set_ylabel('Position Range')
        ax4.set_xticks(head_indices)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_attention_flow_visualization(self, attention_weights: torch.Tensor, 
                                          tokens: list, save_path: str = None):
        """
        创建注意力流可视化
        
        Args:
            attention_weights: 注意力权重张量 [batch_size, n_heads, seq_len, seq_len]
            tokens: token列表
            save_path: 保存路径
        """
        # 选择第一个样本和第一个头
        attention = attention_weights[0, 0].detach().cpu().numpy()
        
        # 创建有向图
        
        G = nx.DiGraph()
        
        # 添加节点
        for i, token in enumerate(tokens):
            G.add_node(i, label=token)
        
        # 添加边（只添加权重大于阈值的边）
        threshold = 0.1
        for i in range(len(tokens)):
            for j in range(len(tokens)):
                if attention[i, j] > threshold:
                    G.add_edge(i, j, weight=attention[i, j])
        
        # 绘制图
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # 绘制节点
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                             node_size=1000, alpha=0.7)
        
        # 绘制边
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        nx.draw_networkx_edges(G, pos, width=weights, 
                             edge_color='gray', alpha=0.6, 
                             arrows=True, arrowsize=20)
        
        # 添加标签
        labels = {node: G.nodes[node]['label'] for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=10)
        
        plt.title('Attention Flow Visualization')
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show() 