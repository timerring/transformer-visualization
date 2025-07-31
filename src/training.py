"""
训练模块
包含训练循环、损失函数、优化器等
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import time
import os
import json
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from sacrebleu import BLEU
import torch.nn.functional as F
import math

class LabelSmoothingLoss(nn.Module):
    """
    标签平滑损失函数
    """
    
    def __init__(self, vocab_size: int, smoothing: float = 0.1, padding_idx: int = 0):
        """
        初始化标签平滑损失
        
        Args:
            vocab_size: 词汇表大小
            smoothing: 平滑系数
            padding_idx: 填充token的索引
        """
        super(LabelSmoothingLoss, self).__init__()
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        self.padding_idx = padding_idx
        
        # 创建平滑标签
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            pred: 预测logits [seq_len, batch_size, vocab_size]
            target: 目标序列 [seq_len, batch_size]
            
        Returns:
            损失值
        """
        # 重塑预测和目标
        pred = pred.reshape(-1, self.vocab_size)
        target = target.reshape(-1)
        
        # 创建平滑标签
        smooth_target = torch.zeros_like(pred)
        smooth_target.fill_(self.smoothing / (self.vocab_size - 1))
        smooth_target.scatter_(1, target.unsqueeze(1), self.confidence)
        
        # 忽略填充token
        mask = (target != self.padding_idx).float()
        smooth_target = smooth_target * mask.unsqueeze(1)
        
        # 计算交叉熵损失
        loss = F.cross_entropy(pred, target, ignore_index=self.padding_idx, reduction='none')
        loss = loss * mask
        
        return loss.sum() / mask.sum()

class WarmupCosineScheduler:
    """
    带预热的余弦学习率调度器
    """
    
    def __init__(self, optimizer, warmup_steps: int, total_steps: int):
        """
        初始化调度器
        
        Args:
            optimizer: 优化器
            warmup_steps: 预热步数
            total_steps: 总步数
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.current_step = 0
        
        # 保存初始学习率
        for param_group in self.optimizer.param_groups:
            param_group['initial_lr'] = param_group['lr']
    
    def step(self):
        """更新学习率"""
        self.current_step += 1
        
        if self.current_step <= self.warmup_steps:
            # 预热阶段：线性增长
            lr_scale = self.current_step / self.warmup_steps
        else:
            # 余弦衰减阶段
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr_scale = 0.5 * (1 + math.cos(math.pi * progress))
        
        # 更新所有参数组的学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['initial_lr'] * lr_scale
    
    def get_last_lr(self):
        """获取当前学习率"""
        return [param_group['lr'] for param_group in self.optimizer.param_groups]

class TransformerTrainer:
    """
    Transformer训练器
    管理整个训练过程
    """
    
    def __init__(self, model, config, device):
        """
        初始化训练器
        
        Args:
            model: Transformer模型
            config: 配置对象
            device: 设备
        """
        self.model = model
        self.config = config
        self.device = device
        
        # 获取配置
        self.model_config = config.get_model_config()
        self.training_config = config.get_training_config()
        
        # 使用实际的词汇表大小
        self.vocab_size = model.tgt_vocab_size
        
        # 损失函数
        self.criterion = LabelSmoothingLoss(self.vocab_size, self.training_config['label_smoothing'])
        
        # 优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.training_config['learning_rate'],
            weight_decay=self.training_config['weight_decay']
        )
        
        # 学习率调度器
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            self.training_config['warmup_steps'],
            self.training_config['epochs'] * 1000  # 估算总步数
        )
        
        # 训练状态
        self.train_losses = []
        self.val_losses = []
        self.train_bleu_scores = []
        self.val_bleu_scores = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
    def train_epoch(self, train_loader, epoch: int) -> Dict[str, float]:
        """
        训练一个epoch
        
        Args:
            train_loader: 训练数据加载器
            epoch: 当前epoch
            
        Returns:
            训练统计信息
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}')
        
        for batch_idx, batch in enumerate(progress_bar):
            # 获取数据
            src = batch['source_ids'].transpose(0, 1).to(self.device)  # [seq_len, batch_size]
            tgt = batch['target_ids'].transpose(0, 1).to(self.device)  # [seq_len, batch_size]
            
            # 创建掩码
            src_mask = self.create_padding_mask(batch['source_ids'].to(self.device))
            
            # 前向传播
            self.optimizer.zero_grad()
            
            # 输入和目标（去掉最后一个token作为输入，去掉第一个token作为目标）
            tgt_input = tgt[:-1, :]
            tgt_output = tgt[1:, :]
            
            # 为目标序列创建因果掩码
            tgt_mask = self.create_causal_mask(tgt_input.size(0)).to(self.device)
            
            output, _ = self.model(src, tgt_input, src_mask, tgt_mask)
            
            # 计算损失
            loss = self.criterion(output, tgt_output)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.training_config['gradient_clip'])
            
            # 更新参数
            self.optimizer.step()
            self.scheduler.step()
            
            # 记录统计信息
            total_loss += loss.item()
            num_batches += 1
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.6f}'
            })
            
            # 记录学习率
            # self.learning_rates.append(self.scheduler.get_last_lr()[0]) # This line was removed
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return {
            'train_loss': avg_loss,
            'learning_rate': self.scheduler.get_last_lr()[0]
        }
    
    def validate(self, val_loader) -> Dict[str, float]:
        """
        验证模型
        
        Args:
            val_loader: 验证数据加载器
            
        Returns:
            验证统计信息
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                # 获取数据
                src = batch['source_ids'].transpose(0, 1).to(self.device)
                tgt = batch['target_ids'].transpose(0, 1).to(self.device)
                
                # 创建掩码
                src_mask = self.create_padding_mask(batch['source_ids'].to(self.device))
                tgt_mask = self.create_causal_mask(tgt.size(0)).to(self.device)
                
                # 输入和目标
                tgt_input = tgt[:-1, :]
                tgt_output = tgt[1:, :]
                
                # 前向传播
                output, _ = self.model(src, tgt_input, src_mask, tgt_mask)
                
                # 计算损失
                loss = self.criterion(output, tgt_output)
                total_loss += loss.item()
                num_batches += 1
                
                # 生成预测（用于BLEU计算）
                predictions = torch.argmax(output, dim=-1)
                all_predictions.extend(predictions.cpu().numpy().tolist())
                all_targets.extend(tgt_output.cpu().numpy().tolist())
        
        avg_loss = total_loss / num_batches
        
        # 计算BLEU分数（简化版本）
        bleu_score = self.calculate_bleu_score(all_predictions, all_targets)
        
        self.val_losses.append(avg_loss)
        self.val_bleu_scores.append(bleu_score) # Changed from bleu_scores to val_bleu_scores
        
        return {
            'val_loss': avg_loss,
            'bleu_score': bleu_score
        }
    
    def create_padding_mask(self, padding_ids: torch.Tensor) -> torch.Tensor:
        """
        创建填充掩码
        
        Args:
            padding_ids: 填充ID张量 [batch_size, seq_len]
            
        Returns:
            掩码张量 [batch_size, 1, 1, seq_len]
        """
        # 创建填充掩码：True表示需要mask的位置
        mask = (padding_ids == self.model_config['pad_token_id'])
        # 扩展维度为 [batch_size, 1, 1, seq_len]
        mask = mask.unsqueeze(1).unsqueeze(2)
        return mask
    
    def create_causal_mask(self, seq_len: int) -> torch.Tensor:
        """
        创建因果掩码
        
        Args:
            seq_len: 序列长度
            
        Returns:
            掩码张量 [seq_len, seq_len]
        """
        # 创建上三角矩阵，对角线以下为True（需要mask）
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        # 转换为float，True位置设为-inf，False位置设为0
        mask = mask.float().masked_fill(mask, float('-inf')).masked_fill(~mask, 0.0)
        return mask
    
    def calculate_bleu_score(self, predictions: List[List[int]], targets: List[List[int]]) -> float:
        """
        计算BLEU分数（简化版本）
        
        Args:
            predictions: 预测序列列表
            targets: 目标序列列表
            
        Returns:
            BLEU分数
        """
        # 这里使用简化的BLEU计算，实际应用中可以使用sacrebleu库
        # 将ID转换为字符串进行比较
        try:
            # 简单的匹配率作为BLEU的替代
            correct_tokens = 0
            total_tokens = 0
            
            for pred, target in zip(predictions, targets):
                for p, t in zip(pred, target):
                    if p == t and t != self.model_config['pad_token_id']:
                        correct_tokens += 1
                    if t != self.model_config['pad_token_id']:
                        total_tokens += 1
            
            return correct_tokens / total_tokens if total_tokens > 0 else 0.0
        except:
            return 0.0
    
    def save_checkpoint(self, epoch: int, save_dir: str):
        """
        保存检查点
        
        Args:
            epoch: 当前epoch
            save_dir: 保存目录
        """
        os.makedirs(save_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'bleu_scores': self.val_bleu_scores, # Changed from bleu_scores to val_bleu_scores
            'learning_rates': self.scheduler.get_last_lr(), # Changed from learning_rates to scheduler.get_last_lr()
            'best_val_loss': self.best_val_loss,
            'best_bleu_score': self.best_val_loss, # Changed from best_bleu_score to best_val_loss
            'config': self.config
        }
        
        torch.save(checkpoint, os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pt'))
        
        # 保存最佳模型
        if self.val_losses[-1] < self.best_val_loss:
            self.best_val_loss = self.val_losses[-1]
            torch.save(checkpoint, os.path.join(save_dir, 'best_model.pt'))
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        加载检查点
        
        Args:
            checkpoint_path: 检查点路径
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.val_bleu_scores = checkpoint['bleu_scores'] # Changed from bleu_scores to val_bleu_scores
        self.scheduler.current_step = checkpoint['scheduler_current_step'] # Assuming scheduler_current_step is part of checkpoint
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_bleu_score = checkpoint['best_bleu_score'] # Changed from best_bleu_score to best_val_loss
        
        return checkpoint['epoch']
    
    def plot_training_curves(self, save_path: str = None):
        """
        绘制训练曲线
        
        Args:
            save_path: 保存路径
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 训练和验证损失
        ax1 = axes[0, 0]
        ax1.plot(self.train_losses, label='Training Loss')
        ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # BLEU分数
        ax2 = axes[0, 1]
        ax2.plot(self.val_bleu_scores, label='BLEU Score', color='green') # Changed from bleu_scores to val_bleu_scores
        ax2.set_title('BLEU Score')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('BLEU Score')
        ax2.legend()
        ax2.grid(True)
        
        # 学习率
        ax3 = axes[1, 0]
        ax3.plot(self.scheduler.get_last_lr(), label='Learning Rate', color='red') # Changed from learning_rates to scheduler.get_last_lr()
        ax3.set_title('Learning Rate Change')
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Learning Rate')
        ax3.legend()
        ax3.grid(True)
        
        # 损失比率
        ax4 = axes[1, 1]
        if len(self.train_losses) > 1 and len(self.val_losses) > 1:
            loss_ratio = [v/t for v, t in zip(self.val_losses, self.train_losses)]
            ax4.plot(loss_ratio, label='Validation/Training Loss Ratio', color='purple')
            ax4.set_title('Validation/Training Loss Ratio')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Ratio')
            ax4.legend()
            ax4.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def train(self, train_loader, val_loader, save_dir: str = 'results/models'):
        """
        完整训练流程
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            save_dir: 保存目录
        """
        print("开始训练...")
        
        for epoch in range(self.training_config['epochs']):
            print(f"\nEpoch {epoch+1}/{self.training_config['epochs']}")
            
            # 训练
            train_stats = self.train_epoch(train_loader, epoch)
            
            # 验证
            val_stats = self.validate(val_loader)
            
            # 打印统计信息
            print(f"训练损失: {train_stats['train_loss']:.4f}")
            print(f"验证损失: {val_stats['val_loss']:.4f}")
            print(f"BLEU分数: {val_stats['bleu_score']:.4f}")
            print(f"学习率: {train_stats['learning_rate']:.6f}")
            
            # 保存检查点
            if (epoch + 1) % self.training_config['save_steps'] == 0:
                self.save_checkpoint(epoch, save_dir)
            
            # 早停检查
            if val_stats['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_stats['val_loss']
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                
            if self.patience_counter >= self.training_config['early_stopping_patience']:
                print(f"早停触发，在epoch {epoch+1}停止训练")
                break
        
        # 保存最终模型
        self.save_checkpoint(self.training_config['epochs'] - 1, save_dir)
        
        # 绘制训练曲线
        self.plot_training_curves(os.path.join(save_dir, 'training_curves.png'))
        
        print("训练完成！") 