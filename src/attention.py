"""
注意力机制模块
实现Transformer中的多头自注意力机制
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple

class MultiHeadAttention(nn.Module):
    """
    多头自注意力机制
    实现Transformer论文中的多头注意力
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        """
        初始化多头注意力
        
        Args:
            d_model: 模型维度
            n_heads: 注意力头数
            dropout: Dropout率
        """
        super(MultiHeadAttention, self).__init__()
        
        assert d_model % n_heads == 0, "d_model必须能被n_heads整除"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # 线性变换层
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        self.attention_dropout = nn.Dropout(dropout)
        
        # 缩放因子
        self.scale = math.sqrt(self.d_k)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            query: 查询张量 [seq_len_q, batch_size, d_model]
            key: 键张量 [seq_len_k, batch_size, d_model]
            value: 值张量 [seq_len_v, batch_size, d_model]
            mask: 掩码张量 [seq_len_q, seq_len_k] 或 [batch_size, n_heads, seq_len_q, seq_len_k]
            
        Returns:
            output: 输出张量 [seq_len_q, batch_size, d_model]
            attention_weights: 注意力权重 [batch_size, n_heads, seq_len_q, seq_len_k]
        """
        batch_size = query.size(1)
        seq_len_q = query.size(0)
        seq_len_k = key.size(0)
        seq_len_v = value.size(0)
        
        # 确保key和value长度相同
        if seq_len_k != seq_len_v:
            raise ValueError(f"Key and value must have the same sequence length, got {seq_len_k} and {seq_len_v}")
        
        # 线性变换并重塑为多头
        Q = self.w_q(query).view(seq_len_q, batch_size, self.n_heads, self.d_k).transpose(0, 1)
        K = self.w_k(key).view(seq_len_k, batch_size, self.n_heads, self.d_k).transpose(0, 1)
        V = self.w_v(value).view(seq_len_v, batch_size, self.n_heads, self.d_k).transpose(0, 1)
        
        # 确保维度顺序正确：[batch_size, n_heads, seq_len, d_k]
        Q = Q.transpose(1, 2)  # [batch_size, n_heads, seq_len_q, d_k]
        K = K.transpose(1, 2)  # [batch_size, n_heads, seq_len_k, d_k]
        V = V.transpose(1, 2)  # [batch_size, n_heads, seq_len_v, d_k]
        
        # 计算注意力权重
        attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # 应用注意力权重
        output = torch.matmul(attention_weights, V)
        
        # 重塑并线性变换
        output = output.transpose(0, 1).contiguous().view(seq_len_q, batch_size, self.d_model)
        output = self.w_o(output)
        output = self.dropout(output)
        
        return output, attention_weights
    
    def scaled_dot_product_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                                   mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        缩放点积注意力
        
        Args:
            Q: 查询张量 [batch_size, n_heads, seq_len_q, d_k]
            K: 键张量 [batch_size, n_heads, seq_len_k, d_k]
            V: 值张量 [batch_size, n_heads, seq_len_v, d_k]
            mask: 掩码张量
            
        Returns:
            注意力权重 [batch_size, n_heads, seq_len_q, seq_len_k]
        """
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [batch_size, n_heads, seq_len_q, seq_len_k]
        
        # 应用掩码
        if mask is not None:
            # 确保掩码维度正确
            if mask.dim() == 2:
                # [seq_len_q, seq_len_k] -> [1, 1, seq_len_q, seq_len_k]
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                # [batch_size, 1, seq_len] -> [batch_size, 1, 1, seq_len]
                mask = mask.unsqueeze(1)
            elif mask.dim() == 4:
                # [batch_size, 1, 1, seq_len] -> [batch_size, 1, seq_len_q, seq_len_k]
                if mask.size(2) == 1:
                    mask = mask.expand(-1, -1, scores.size(2), -1)
            
            # 确保掩码维度与scores匹配
            if mask.size(-1) != scores.size(-1) or mask.size(-2) != scores.size(-2):
                # 如果序列长度不匹配，调整掩码
                target_size_k = scores.size(-1)
                target_size_q = scores.size(-2)
                if mask.size(-1) > target_size_k:
                    mask = mask[:, :, :target_size_q, :target_size_k]
                else:
                    # 扩展掩码到目标大小
                    new_mask = torch.zeros(mask.size(0), mask.size(1), target_size_q, target_size_k, 
                                         device=mask.device, dtype=mask.dtype)
                    new_mask[:, :, :mask.size(-2), :mask.size(-1)] = mask
                    mask = new_mask
            
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 应用softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)
        
        return attention_weights

class SelfAttention(nn.Module):
    """
    自注意力机制
    对同一序列进行注意力计算
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        """
        初始化自注意力
        
        Args:
            d_model: 模型维度
            n_heads: 注意力头数
            dropout: Dropout率
        """
        super(SelfAttention, self).__init__()
        self.multi_head_attention = MultiHeadAttention(d_model, n_heads, dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入张量 [seq_len, batch_size, d_model]
            mask: 掩码张量
            
        Returns:
            output: 输出张量 [seq_len, batch_size, d_model]
            attention_weights: 注意力权重
        """
        return self.multi_head_attention(x, x, x, mask)

class CrossAttention(nn.Module):
    """
    交叉注意力机制
    用于编码器-解码器之间的注意力
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        """
        初始化交叉注意力
        
        Args:
            d_model: 模型维度
            n_heads: 注意力头数
            dropout: Dropout率
        """
        super(CrossAttention, self).__init__()
        self.multi_head_attention = MultiHeadAttention(d_model, n_heads, dropout)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            query: 查询张量（来自解码器）[seq_len, batch_size, d_model]
            key: 键张量（来自编码器）[seq_len, batch_size, d_model]
            value: 值张量（来自编码器）[seq_len, batch_size, d_model]
            mask: 掩码张量
            
        Returns:
            output: 输出张量 [seq_len, batch_size, d_model]
            attention_weights: 注意力权重
        """
        return self.multi_head_attention(query, key, value, mask)

def create_attention_mask(seq_len: int, device: torch.device = None) -> torch.Tensor:
    """
    创建因果掩码（用于解码器）
    
    Args:
        seq_len: 序列长度
        device: 设备
        
    Returns:
        掩码张量 [seq_len, seq_len]
    """
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    mask = mask.masked_fill(mask == 0, float(0.0))
    
    if device:
        mask = mask.to(device)
    
    return mask

def create_padding_mask(padding_ids: torch.Tensor) -> torch.Tensor:
    """
    创建填充掩码
    
    Args:
        padding_ids: 填充ID张量 [batch_size, seq_len]
        
    Returns:
        掩码张量 [batch_size, 1, 1, seq_len]
    """
    mask = (padding_ids == 0).unsqueeze(1).unsqueeze(2)
    return mask 