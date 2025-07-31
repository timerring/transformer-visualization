"""
Transformer模型模块
实现完整的Transformer架构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, Any
from .attention import MultiHeadAttention, SelfAttention, CrossAttention
from .positional_encoding import PositionalEncoding

class FeedForward(nn.Module):
    """
    前馈神经网络
    实现Transformer中的前馈层
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        初始化前馈网络
        
        Args:
            d_model: 模型维度
            d_ff: 前馈网络维度
            dropout: Dropout率
        """
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [seq_len, batch_size, d_model]
            
        Returns:
            输出张量 [seq_len, batch_size, d_model]
        """
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class EncoderLayer(nn.Module):
    """
    编码器层
    包含多头自注意力和前馈网络
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1,
                 use_residual: bool = True, use_layer_norm: bool = True):
        """
        初始化编码器层
        
        Args:
            d_model: 模型维度
            n_heads: 注意力头数
            d_ff: 前馈网络维度
            dropout: Dropout率
            use_residual: 是否使用残差连接
            use_layer_norm: 是否使用层归一化
        """
        super(EncoderLayer, self).__init__()
        
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm
        
        # 多头自注意力
        self.self_attention = SelfAttention(d_model, n_heads, dropout)
        
        # 前馈网络
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        # 层归一化
        if use_layer_norm:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
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
        # 多头自注意力
        attn_output, attention_weights = self.self_attention(x, mask)
        
        # 残差连接和层归一化
        if self.use_residual:
            x = x + self.dropout(attn_output)
        else:
            x = self.dropout(attn_output)
            
        if self.use_layer_norm:
            x = self.norm1(x)
        
        # 前馈网络
        ff_output = self.feed_forward(x)
        
        # 残差连接和层归一化
        if self.use_residual:
            x = x + self.dropout(ff_output)
        else:
            x = self.dropout(ff_output)
            
        if self.use_layer_norm:
            x = self.norm2(x)
        
        return x, attention_weights

class DecoderLayer(nn.Module):
    """
    解码器层
    包含多头自注意力、交叉注意力和前馈网络
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1,
                 use_residual: bool = True, use_layer_norm: bool = True):
        """
        初始化解码器层
        
        Args:
            d_model: 模型维度
            n_heads: 注意力头数
            d_ff: 前馈网络维度
            dropout: Dropout率
            use_residual: 是否使用残差连接
            use_layer_norm: 是否使用层归一化
        """
        super(DecoderLayer, self).__init__()
        
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm
        
        # 多头自注意力（带掩码）
        self.self_attention = SelfAttention(d_model, n_heads, dropout)
        
        # 交叉注意力
        self.cross_attention = CrossAttention(d_model, n_heads, dropout)
        
        # 前馈网络
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        # 层归一化
        if use_layer_norm:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.norm3 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入张量 [seq_len, batch_size, d_model]
            encoder_output: 编码器输出 [seq_len, batch_size, d_model]
            src_mask: 源序列掩码
            tgt_mask: 目标序列掩码
            
        Returns:
            output: 输出张量 [seq_len, batch_size, d_model]
            self_attention_weights: 自注意力权重
            cross_attention_weights: 交叉注意力权重
        """
        # 多头自注意力（带掩码）
        attn_output, self_attention_weights = self.self_attention(x, tgt_mask)
        
        # 残差连接和层归一化
        if self.use_residual:
            x = x + self.dropout(attn_output)
        else:
            x = self.dropout(attn_output)
            
        if self.use_layer_norm:
            x = self.norm1(x)
        
        # 交叉注意力
        cross_output, cross_attention_weights = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        
        # 残差连接和层归一化
        if self.use_residual:
            x = x + self.dropout(cross_output)
        else:
            x = self.dropout(cross_output)
            
        if self.use_layer_norm:
            x = self.norm2(x)
        
        # 前馈网络
        ff_output = self.feed_forward(x)
        
        # 残差连接和层归一化
        if self.use_residual:
            x = x + self.dropout(ff_output)
        else:
            x = self.dropout(ff_output)
            
        if self.use_layer_norm:
            x = self.norm3(x)
        
        return x, self_attention_weights, cross_attention_weights

class Encoder(nn.Module):
    """
    编码器
    包含多个编码器层
    """
    
    def __init__(self, d_model: int, n_heads: int, n_layers: int, d_ff: int, 
                 dropout: float = 0.1, use_residual: bool = True, use_layer_norm: bool = True):
        """
        初始化编码器
        
        Args:
            d_model: 模型维度
            n_heads: 注意力头数
            n_layers: 层数
            d_ff: 前馈网络维度
            dropout: Dropout率
            use_residual: 是否使用残差连接
            use_layer_norm: 是否使用层归一化
        """
        super(Encoder, self).__init__()
        
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout, use_residual, use_layer_norm)
            for _ in range(n_layers)
        ])
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, list]:
        """
        前向传播
        
        Args:
            x: 输入张量 [seq_len, batch_size, d_model]
            mask: 掩码张量
            
        Returns:
            output: 输出张量 [seq_len, batch_size, d_model]
            attention_weights: 所有层的注意力权重列表
        """
        attention_weights = []
        
        for layer in self.layers:
            x, attn_weights = layer(x, mask)
            attention_weights.append(attn_weights)
        
        return x, attention_weights

class Decoder(nn.Module):
    """
    解码器
    包含多个解码器层
    """
    
    def __init__(self, d_model: int, n_heads: int, n_layers: int, d_ff: int,
                 dropout: float = 0.1, use_residual: bool = True, use_layer_norm: bool = True):
        """
        初始化解码器
        
        Args:
            d_model: 模型维度
            n_heads: 注意力头数
            n_layers: 层数
            d_ff: 前馈网络维度
            dropout: Dropout率
            use_residual: 是否使用残差连接
            use_layer_norm: 是否使用层归一化
        """
        super(Decoder, self).__init__()
        
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout, use_residual, use_layer_norm)
            for _ in range(n_layers)
        ])
        
    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, list, list]:
        """
        前向传播
        
        Args:
            x: 输入张量 [seq_len, batch_size, d_model]
            encoder_output: 编码器输出 [seq_len, batch_size, d_model]
            src_mask: 源序列掩码
            tgt_mask: 目标序列掩码
            
        Returns:
            output: 输出张量 [seq_len, batch_size, d_model]
            self_attention_weights: 所有层的自注意力权重列表
            cross_attention_weights: 所有层的交叉注意力权重列表
        """
        self_attention_weights = []
        cross_attention_weights = []
        
        for layer in self.layers:
            x, self_attn, cross_attn = layer(x, encoder_output, src_mask, tgt_mask)
            self_attention_weights.append(self_attn)
            cross_attention_weights.append(cross_attn)
        
        return x, self_attention_weights, cross_attention_weights

class Transformer(nn.Module):
    """
    完整的Transformer模型
    包含编码器、解码器和必要的组件
    """
    
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, d_model: int = 512,
                 n_heads: int = 8, n_layers: int = 6, d_ff: int = 2048,
                 max_seq_length: int = 5000, dropout: float = 0.1,
                 use_residual: bool = True, use_layer_norm: bool = True,
                 use_positional_encoding: bool = True):
        """
        初始化Transformer模型
        
        Args:
            src_vocab_size: 源语言词汇表大小
            tgt_vocab_size: 目标语言词汇表大小
            d_model: 模型维度
            n_heads: 注意力头数
            n_layers: 层数
            d_ff: 前馈网络维度
            max_seq_length: 最大序列长度
            dropout: Dropout率
            use_residual: 是否使用残差连接
            use_layer_norm: 是否使用层归一化
            use_positional_encoding: 是否使用位置编码
        """
        super(Transformer, self).__init__()
        
        self.d_model = d_model
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.use_positional_encoding = use_positional_encoding
        
        # 词嵌入层
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        # 位置编码
        if use_positional_encoding:
            self.positional_encoding = PositionalEncoding(d_model, max_seq_length, dropout)
        
        # 编码器和解码器
        self.encoder = Encoder(d_model, n_heads, n_layers, d_ff, dropout, use_residual, use_layer_norm)
        self.decoder = Decoder(d_model, n_heads, n_layers, d_ff, dropout, use_residual, use_layer_norm)
        
        # 输出层
        self.output_layer = nn.Linear(d_model, tgt_vocab_size)
        
        # 初始化参数
        self._init_parameters()
        
    def _init_parameters(self):
        """初始化模型参数"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, list]:
        """
        编码源序列
        
        Args:
            src: 源序列 [seq_len, batch_size]
            src_mask: 源序列掩码
            
        Returns:
            encoder_output: 编码器输出 [seq_len, batch_size, d_model]
            attention_weights: 注意力权重
        """
        # 词嵌入
        src_embedded = self.src_embedding(src) * math.sqrt(self.d_model)
        
        # 位置编码
        if self.use_positional_encoding:
            src_embedded = self.positional_encoding(src_embedded)
        
        # 编码
        encoder_output, attention_weights = self.encoder(src_embedded, src_mask)
        
        return encoder_output, attention_weights
    
    def decode(self, tgt: torch.Tensor, encoder_output: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, list, list]:
        """
        解码目标序列
        
        Args:
            tgt: 目标序列 [seq_len, batch_size]
            encoder_output: 编码器输出 [seq_len, batch_size, d_model]
            src_mask: 源序列掩码
            tgt_mask: 目标序列掩码
            
        Returns:
            decoder_output: 解码器输出 [seq_len, batch_size, d_model]
            self_attention_weights: 自注意力权重
            cross_attention_weights: 交叉注意力权重
        """
        # 词嵌入
        tgt_embedded = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        
        # 位置编码
        if self.use_positional_encoding:
            tgt_embedded = self.positional_encoding(tgt_embedded)
        
        # 解码
        decoder_output, self_attention_weights, cross_attention_weights = self.decoder(
            tgt_embedded, encoder_output, src_mask, tgt_mask
        )
        
        return decoder_output, self_attention_weights, cross_attention_weights
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        前向传播
        
        Args:
            src: 源序列 [seq_len, batch_size]
            tgt: 目标序列 [seq_len, batch_size]
            src_mask: 源序列掩码
            tgt_mask: 目标序列掩码
            
        Returns:
            output: 输出logits [seq_len, batch_size, tgt_vocab_size]
            attention_info: 注意力信息字典
        """
        # 编码
        encoder_output, encoder_attention_weights = self.encode(src, src_mask)
        
        # 解码
        decoder_output, self_attention_weights, cross_attention_weights = self.decode(
            tgt, encoder_output, src_mask, tgt_mask
        )
        
        # 输出层
        output = self.output_layer(decoder_output)
        
        # 收集注意力信息
        attention_info = {
            'encoder_attention_weights': encoder_attention_weights,
            'decoder_self_attention_weights': self_attention_weights,
            'decoder_cross_attention_weights': cross_attention_weights
        }
        
        return output, attention_info
    
    def generate(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None,
                 max_length: int = 50, sos_token_id: int = 2, eos_token_id: int = 3,
                 pad_token_id: int = 0) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        生成翻译序列
        
        Args:
            src: 源序列 [seq_len, batch_size]
            src_mask: 源序列掩码
            max_length: 最大生成长度
            sos_token_id: 开始token ID
            eos_token_id: 结束token ID
            pad_token_id: 填充token ID
            
        Returns:
            generated_sequence: 生成的序列 [max_length, batch_size]
            attention_info: 注意力信息
        """
        batch_size = src.size(1)
        device = src.device
        
        # 编码
        encoder_output, encoder_attention_weights = self.encode(src, src_mask)
        
        # 初始化生成序列
        generated_sequence = torch.full((max_length, batch_size), pad_token_id, 
                                      dtype=torch.long, device=device)
        generated_sequence[0, :] = sos_token_id
        
        # 收集注意力信息
        all_self_attention_weights = []
        all_cross_attention_weights = []
        
        # 自回归生成
        for i in range(1, max_length):
            # 创建目标序列（只包含已生成的部分）
            tgt = generated_sequence[:i, :]
            
            # 创建因果掩码
            tgt_mask = torch.triu(torch.ones(i, i), diagonal=1).bool().to(device)
            tgt_mask = tgt_mask.masked_fill(tgt_mask, float('-inf'))
            tgt_mask = tgt_mask.masked_fill(~tgt_mask, float(0.0))
            
            # 解码
            decoder_output, self_attention_weights, cross_attention_weights = self.decode(
                tgt, encoder_output, src_mask, tgt_mask
            )
            
            # 获取下一个token的预测
            next_token_logits = self.output_layer(decoder_output[-1, :, :])  # [batch_size, vocab_size]
            next_token = torch.argmax(next_token_logits, dim=-1)  # [batch_size]
            
            # 更新生成序列
            generated_sequence[i, :] = next_token
            
            # 收集注意力权重
            all_self_attention_weights.append(self_attention_weights)
            all_cross_attention_weights.append(cross_attention_weights)
            
            # 检查是否所有序列都生成了结束token
            if (next_token == eos_token_id).all():
                break
        
        attention_info = {
            'encoder_attention_weights': encoder_attention_weights,
            'decoder_self_attention_weights': all_self_attention_weights,
            'decoder_cross_attention_weights': all_cross_attention_weights
        }
        
        return generated_sequence, attention_info 