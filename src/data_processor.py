"""
数据处理器模块
包含数据加载、预处理、词汇表构建等功能
"""

import os
import json
import random
import re
from collections import Counter
from typing import List, Tuple, Dict
import torch
from torch.utils.data import DataLoader
from .data_utils import TranslationDataset

class DataProcessor:
    """数据处理器类"""
    
    def __init__(self, config):
        """
        初始化数据处理器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.source_vocab = {}
        self.target_vocab = {}
        self.source_vocab_reverse = {}
        self.target_vocab_reverse = {}
        
    def load_data(self, file_path: str) -> Tuple[List[str], List[str]]:
        """
        加载翻译数据
        
        Args:
            file_path: 数据文件路径
            
        Returns:
            source_texts, target_texts: 源语言和目标语言文本列表
        """
        source_texts = []
        target_texts = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if '\t' in line:
                    source, target = line.split('\t', 1)
                    source_texts.append(source.strip())
                    target_texts.append(target.strip())
        
        return source_texts, target_texts
    
    def preprocess_text(self, text: str) -> str:
        """
        文本预处理
        
        Args:
            text: 原始文本
            
        Returns:
            预处理后的文本
        """
        # 转换为小写
        text = text.lower()
        
        # 移除特殊字符，保留基本标点
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
        
        # 规范化空白字符
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def build_vocabulary(self, texts: List[str], min_freq: int = 2) -> Dict[str, int]:
        """
        构建词汇表
        
        Args:
            texts: 文本列表
            min_freq: 最小词频
            
        Returns:
            词汇表字典
        """
        # 统计词频
        word_counts = Counter()
        for text in texts:
            tokens = text.split()
            word_counts.update(tokens)
        
        # 过滤低频词
        vocab = {'<pad>': 0, '<unk>': 1, '<sos>': 2, '<eos>': 3}
        for word, count in word_counts.items():
            if count >= min_freq:
                vocab[word] = len(vocab)
        
        return vocab
    
    def create_datasets(self, source_texts: List[str], target_texts: List[str],
                       train_ratio: float = 0.8, val_ratio: float = 0.1) -> Dict[str, TranslationDataset]:
        """
        创建训练、验证和测试数据集
        
        Args:
            source_texts: 源语言文本列表
            target_texts: 目标语言文本列表
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            
        Returns:
            数据集字典
        """
        # 预处理文本
        processed_source = [self.preprocess_text(text) for text in source_texts]
        processed_target = [self.preprocess_text(text) for text in target_texts]
        
        # 构建词汇表
        self.source_vocab = self.build_vocabulary(processed_source, self.config.get_data_config()['min_freq'])
        self.target_vocab = self.build_vocabulary(processed_target, self.config.get_data_config()['min_freq'])
        
        # 创建反向词汇表
        self.source_vocab_reverse = {v: k for k, v in self.source_vocab.items()}
        self.target_vocab_reverse = {v: k for k, v in self.target_vocab.items()}
        
        # 划分数据集
        total_size = len(processed_source)
        train_size = int(total_size * train_ratio)
        val_size = int(total_size * val_ratio)
        
        # 随机打乱数据
        indices = list(range(total_size))
        random.shuffle(indices)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        # 创建数据集
        datasets = {}
        
        # 训练集
        train_source = [processed_source[i] for i in train_indices]
        train_target = [processed_target[i] for i in train_indices]
        datasets['train'] = TranslationDataset(
            train_source, train_target, self.source_vocab, self.target_vocab,
            self.config.get_data_config()['max_length']
        )
        
        # 验证集
        val_source = [processed_source[i] for i in val_indices]
        val_target = [processed_target[i] for i in val_indices]
        datasets['val'] = TranslationDataset(
            val_source, val_target, self.source_vocab, self.target_vocab,
            self.config.get_data_config()['max_length']
        )
        
        # 测试集
        test_source = [processed_source[i] for i in test_indices]
        test_target = [processed_target[i] for i in test_indices]
        datasets['test'] = TranslationDataset(
            test_source, test_target, self.source_vocab, self.target_vocab,
            self.config.get_data_config()['max_length']
        )
        
        return datasets
    
    def create_dataloaders(self, datasets: Dict[str, TranslationDataset]) -> Dict[str, DataLoader]:
        """
        创建数据加载器
        
        Args:
            datasets: 数据集字典
            
        Returns:
            数据加载器字典
        """
        dataloaders = {}
        batch_size = self.config.get_data_config()['batch_size']
        num_workers = self.config.get_data_config()['num_workers']
        
        for split, dataset in datasets.items():
            shuffle = (split == 'train')
            dataloaders[split] = DataLoader(
                dataset, batch_size=batch_size, shuffle=shuffle,
                num_workers=num_workers, collate_fn=self.collate_fn
            )
        
        return dataloaders
    
    def collate_fn(self, batch):
        """
        批处理函数
        
        Args:
            batch: 批次数据
            
        Returns:
            处理后的批次数据
        """
        source_ids = torch.stack([item['source_ids'] for item in batch])
        target_ids = torch.stack([item['target_ids'] for item in batch])
        source_lengths = torch.tensor([item['source_length'] for item in batch])
        target_lengths = torch.tensor([item['target_length'] for item in batch])
        
        return {
            'source_ids': source_ids,
            'target_ids': target_ids,
            'source_lengths': source_lengths,
            'target_lengths': target_lengths
        }
    
    def save_vocabularies(self, save_dir: str):
        """保存词汇表"""
        os.makedirs(save_dir, exist_ok=True)
        
        with open(os.path.join(save_dir, 'source_vocab.json'), 'w', encoding='utf-8') as f:
            json.dump(self.source_vocab, f, ensure_ascii=False, indent=2)
        
        with open(os.path.join(save_dir, 'target_vocab.json'), 'w', encoding='utf-8') as f:
            json.dump(self.target_vocab, f, ensure_ascii=False, indent=2)
    
    def load_vocabularies(self, save_dir: str):
        """加载词汇表"""
        with open(os.path.join(save_dir, 'source_vocab.json'), 'r', encoding='utf-8') as f:
            self.source_vocab = json.load(f)
        
        with open(os.path.join(save_dir, 'target_vocab.json'), 'r', encoding='utf-8') as f:
            self.target_vocab = json.load(f)
        
        # 创建反向词汇表
        self.source_vocab_reverse = {v: k for k, v in self.source_vocab.items()}
        self.target_vocab_reverse = {v: k for k, v in self.target_vocab.items()} 