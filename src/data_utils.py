"""
数据处理工具模块
包含数据加载、预处理、批处理等功能
"""

import os
import json
import random
import numpy as np
import pandas as pd
from collections import Counter
from typing import List, Tuple, Dict
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import re

class TranslationDataset(Dataset):
    """机器翻译数据集类"""
    
    def __init__(self, source_texts: List[str], target_texts: List[str], 
                 source_vocab: Dict[str, int], target_vocab: Dict[str, int],
                 max_length: int = 100, pad_token_id: int = 0):
        """
        初始化翻译数据集
        
        Args:
            source_texts: 源语言文本列表
            target_texts: 目标语言文本列表
            source_vocab: 源语言词汇表
            target_vocab: 目标语言词汇表
            max_length: 最大序列长度
            pad_token_id: 填充token的ID
        """
        self.source_texts = source_texts
        self.target_texts = target_texts
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        
        # 特殊token ID
        self.unk_token_id = 1
        self.sos_token_id = 2
        self.eos_token_id = 3
        
    def __len__(self):
        return len(self.source_texts)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        source_text = self.source_texts[idx]
        target_text = self.target_texts[idx]
        
        # 文本转token ID
        source_ids = self.text_to_ids(source_text, self.source_vocab)
        target_ids = self.text_to_ids(target_text, self.target_vocab)
        
        # 添加特殊token
        source_ids = [self.sos_token_id] + source_ids + [self.eos_token_id]
        target_ids = [self.sos_token_id] + target_ids + [self.eos_token_id]
        
        # 截断和填充
        source_ids = self.pad_sequence(source_ids, self.max_length)
        target_ids = self.pad_sequence(target_ids, self.max_length)
        
        return {
            'source_ids': torch.tensor(source_ids, dtype=torch.long),
            'target_ids': torch.tensor(target_ids, dtype=torch.long),
            'source_length': len(source_ids),
            'target_length': len(target_ids)
        }
    
    def text_to_ids(self, text: str, vocab: Dict[str, int]) -> List[int]:
        """将文本转换为token ID序列"""
        tokens = text.split()
        ids = []
        for token in tokens:
            if token in vocab:
                ids.append(vocab[token])
            else:
                ids.append(self.unk_token_id)
        return ids
    
    def pad_sequence(self, sequence: List[int], max_length: int) -> List[int]:
        """填充或截断序列"""
        if len(sequence) > max_length:
            return sequence[:max_length]
        else:
            return sequence + [self.pad_token_id] * (max_length - len(sequence))

def generate_sample_data(num_samples: int = 10000) -> Tuple[List[str], List[str]]:
    """
    生成示例翻译数据
    
    Args:
        num_samples: 样本数量
        
    Returns:
        source_texts, target_texts: 源语言和目标语言文本列表
    """
    # 简单的英文到中文翻译示例
    sample_pairs = [
        ("hello world", "你好 世界"),
        ("how are you", "你好 吗"),
        ("thank you", "谢谢 你"),
        ("good morning", "早上 好"),
        ("good night", "晚安"),
        ("i love you", "我 爱 你"),
        ("what is your name", "你 叫 什么 名字"),
        ("where are you from", "你 来自 哪里"),
        ("how old are you", "你 多 大 了"),
        ("nice to meet you", "很 高兴 认识 你"),
        ("see you later", "再见"),
        ("have a good day", "祝 你 有 美好 的 一天"),
        ("what time is it", "现在 几 点 了"),
        ("i am hungry", "我 饿 了"),
        ("let's go", "我们 走 吧"),
        ("this is beautiful", "这 很 美"),
        ("i understand", "我 明白"),
        ("please help me", "请 帮 我"),
        ("excuse me", "对不起"),
        ("you're welcome", "不 客气")
    ]
    
    source_texts = []
    target_texts = []
    
    for _ in range(num_samples):
        # 随机选择基础句子
        base_source, base_target = random.choice(sample_pairs)
        
        # 添加一些变化
        variations = [
            (base_source, base_target),
            (base_source + "!", base_target + "！"),
            (base_source + "?", base_target + "？"),
            ("the " + base_source, "这个 " + base_target),
            ("my " + base_source, "我的 " + base_target),
            ("your " + base_source, "你的 " + base_target)
        ]
        
        source, target = random.choice(variations)
        source_texts.append(source)
        target_texts.append(target)
    
    return source_texts, target_texts 