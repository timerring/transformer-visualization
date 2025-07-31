"""
位置编码模块
实现Transformer中的位置编码功能
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class PositionalEncoding(nn.Module):
    """
    位置编码类
    实现Transformer论文中的正弦位置编码
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        初始化位置编码
        
        Args:
            d_model: 模型维度
            max_len: 最大序列长度
            dropout: Dropout率
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # 计算位置编码
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        # 偶数位置使用sin
        pe[:, 0::2] = torch.sin(position * div_term)
        # 奇数位置使用cos
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 添加batch维度并注册为buffer（不参与梯度更新）
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 [seq_len, batch_size, d_model]
            
        Returns:
            添加位置编码后的张量
        """
        # 添加位置编码
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class LearnedPositionalEncoding(nn.Module):
    """
    学习的位置编码类
    通过可学习的参数来学习位置信息
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        初始化学习的位置编码
        
        Args:
            d_model: 模型维度
            max_len: 最大序列长度
            dropout: Dropout率
        """
        super(LearnedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 可学习的位置编码
        self.pe = nn.Parameter(torch.randn(max_len, d_model))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 [seq_len, batch_size, d_model]
            
        Returns:
            添加位置编码后的张量
        """
        # 添加位置编码
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class RelativePositionalEncoding(nn.Module):
    """
    相对位置编码类
    实现相对位置编码，考虑token之间的相对距离
    """
    
    def __init__(self, d_model: int, max_relative_position: int = 32, dropout: float = 0.1):
        """
        初始化相对位置编码
        
        Args:
            d_model: 模型维度
            max_relative_position: 最大相对位置
            dropout: Dropout率
        """
        super(RelativePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_relative_position = max_relative_position
        
        # 相对位置编码矩阵
        self.relative_position_encoding = nn.Parameter(
            torch.randn(2 * max_relative_position + 1, d_model)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 [seq_len, batch_size, d_model]
            
        Returns:
            添加相对位置编码后的张量
        """
        seq_len = x.size(0)
        
        # 计算相对位置
        range_vec = torch.arange(seq_len)
        range_mat = range_vec.unsqueeze(0).repeat(seq_len, 1)
        distance_mat = range_mat - range_mat.transpose(0, 1)
        
        # 将相对位置映射到编码索引
        distance_mat_clipped = torch.clamp(
            distance_mat, 
            -self.max_relative_position, 
            self.max_relative_position
        )
        final_mat = distance_mat_clipped + self.max_relative_position
        
        # 获取相对位置编码
        embeddings = self.relative_position_encoding[final_mat]
        
        # 添加相对位置编码
        x = x + embeddings
        return self.dropout(x)

class PositionalEncodingAnalyzer:
    """
    位置编码分析器
    用于分析和可视化位置编码的特性
    """
    
    def __init__(self, d_model: int = 512, max_len: int = 100):
        """
        初始化分析器
        
        Args:
            d_model: 模型维度
            max_len: 最大序列长度
        """
        self.d_model = d_model
        self.max_len = max_len
        
    def analyze_sinusoidal_encoding(self):
        """分析正弦位置编码的特性"""
        # 创建位置编码
        pe = torch.zeros(self.max_len, self.d_model)
        position = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * 
                           (-math.log(10000.0) / self.d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe
    
    def visualize_positional_encoding(self, pe: torch.Tensor, save_path: str = None):
        """
        可视化位置编码
        
        Args:
            pe: 位置编码张量
            save_path: 保存路径
        """
        plt.figure(figsize=(15, 10))
        
        # 创建热力图
        plt.subplot(2, 2, 1)
        sns.heatmap(pe.numpy(), cmap='viridis', cbar=True)
        plt.title('Positional Encoding Heatmap')
        plt.xlabel('Model Dimension')
        plt.ylabel('Position')
        
        # 显示前几个维度的编码
        plt.subplot(2, 2, 2)
        for i in range(min(8, self.d_model)):
            plt.plot(pe[:, i].numpy(), label=f'Dimension {i}')
        plt.title('First 8 Dimensions of Positional Encoding')
        plt.xlabel('Position')
        plt.ylabel('Encoding Value')
        plt.legend()
        
        # 显示不同位置的编码分布
        plt.subplot(2, 2, 3)
        positions = [0, 10, 20, 30, 40, 50]
        for pos in positions:
            if pos < self.max_len:
                plt.plot(pe[pos, :].numpy(), label=f'Position {pos}')
        plt.title('Encoding Distribution at Different Positions')
        plt.xlabel('Model Dimension')
        plt.ylabel('Encoding Value')
        plt.legend()
        
        # 编码值的统计分布
        plt.subplot(2, 2, 4)
        plt.hist(pe.numpy().flatten(), bins=50, alpha=0.7)
        plt.title('Positional Encoding Value Distribution')
        plt.xlabel('Encoding Value')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def analyze_encoding_properties(self, pe: torch.Tensor):
        """
        分析位置编码的性质
        
        Args:
            pe: 位置编码张量
            
        Returns:
            分析结果字典
        """
        # 计算编码的统计特性
        mean_val = torch.mean(pe).item()
        std_val = torch.std(pe).item()
        min_val = torch.min(pe).item()
        max_val = torch.max(pe).item()
        
        # 计算不同位置之间的相似度
        similarity_matrix = torch.mm(pe, pe.t())
        
        # 计算相邻位置的相似度
        adjacent_similarity = []
        for i in range(self.max_len - 1):
            sim = F.cosine_similarity(pe[i:i+1], pe[i+1:i+2], dim=1).item()
            adjacent_similarity.append(sim)
        
        # 计算远距离位置的相似度
        distant_similarity = []
        for i in range(0, self.max_len, 10):
            for j in range(i + 20, min(i + 30, self.max_len)):
                sim = F.cosine_similarity(pe[i:i+1], pe[j:j+1], dim=1).item()
                distant_similarity.append(sim)
        
        return {
            'mean': mean_val,
            'std': std_val,
            'min': min_val,
            'max': max_val,
            'adjacent_similarity_mean': np.mean(adjacent_similarity),
            'distant_similarity_mean': np.mean(distant_similarity),
            'similarity_matrix': similarity_matrix.numpy(),
            'adjacent_similarity': adjacent_similarity,
            'distant_similarity': distant_similarity
        }
    
    def compare_encoding_methods(self):
        """比较不同的位置编码方法"""
        # 正弦位置编码
        sinusoidal_pe = self.analyze_sinusoidal_encoding()
        
        # 学习的位置编码
        learned_pe = torch.randn(self.max_len, self.d_model)
        
        # 分析结果
        sinusoidal_props = self.analyze_encoding_properties(sinusoidal_pe)
        learned_props = self.analyze_encoding_properties(learned_pe)
        
        # 可视化比较
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(sinusoidal_props['adjacent_similarity'], label='Sinusoidal Encoding')
        plt.plot(learned_props['adjacent_similarity'], label='Learned Encoding')
        plt.title('Adjacent Position Similarity Comparison')
        plt.xlabel('Position')
        plt.ylabel('Similarity')
        plt.legend()
        
        plt.subplot(1, 3, 2)
        plt.hist(sinusoidal_props['adjacent_similarity'], alpha=0.7, label='Sinusoidal Encoding')
        plt.hist(learned_props['adjacent_similarity'], alpha=0.7, label='Learned Encoding')
        plt.title('Adjacent Position Similarity Distribution')
        plt.xlabel('Similarity')
        plt.ylabel('Frequency')
        plt.legend()
        
        plt.subplot(1, 3, 3)
        plt.bar(['Sinusoidal Encoding', 'Learned Encoding'], 
                [sinusoidal_props['adjacent_similarity_mean'], 
                 learned_props['adjacent_similarity_mean']])
        plt.title('Average Adjacent Position Similarity')
        plt.ylabel('Similarity')
        
        plt.tight_layout()
        plt.show()
        
        return {
            'sinusoidal': sinusoidal_props,
            'learned': learned_props
        }

def create_positional_encoding(encoding_type: str = 'sinusoidal', **kwargs) -> nn.Module:
    """
    创建位置编码的工厂函数
    
    Args:
        encoding_type: 编码类型 ('sinusoidal', 'learned', 'relative')
        **kwargs: 其他参数
        
    Returns:
        位置编码模块
    """
    if encoding_type == 'sinusoidal':
        return PositionalEncoding(**kwargs)
    elif encoding_type == 'learned':
        return LearnedPositionalEncoding(**kwargs)
    elif encoding_type == 'relative':
        return RelativePositionalEncoding(**kwargs)
    else:
        raise ValueError(f"不支持的位置编码类型: {encoding_type}") 