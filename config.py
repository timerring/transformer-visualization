"""
Transformer实验配置文件
包含所有实验参数、模型配置和训练设置
"""

import torch

class Config:
    """实验配置类"""
    
    # 数据配置
    DATA_CONFIG = {
        'source_lang': 'en',
        'target_lang': 'zh',
        'max_length': 100,
        'min_freq': 2,
        'train_ratio': 0.8,
        'val_ratio': 0.1,
        'test_ratio': 0.1,
        'batch_size': 32,
        'num_workers': 4
    }
    
    # 模型配置
    MODEL_CONFIG = {
        'd_model': 512,           # 模型维度
        'n_heads': 8,             # 注意力头数
        'n_layers': 6,            # 编码器和解码器层数
        'd_ff': 2048,             # 前馈网络维度
        'dropout': 0.1,           # Dropout率
        'max_seq_length': 100,    # 最大序列长度
        'vocab_size': 30000,      # 词汇表大小
        'pad_token_id': 0,        # 填充token ID
        'unk_token_id': 1,        # 未知token ID
        'sos_token_id': 2,        # 开始token ID
        'eos_token_id': 3,        # 结束token ID
    }
    
    # 训练配置
    TRAINING_CONFIG = {
        'epochs': 50,
        'learning_rate': 1e-4,
        'warmup_steps': 4000,
        'weight_decay': 1e-4,
        'gradient_clip': 1.0,
        'label_smoothing': 0.1,
        'scheduler': 'cosine',    # 'cosine', 'linear', 'step'
        'save_steps': 1000,
        'eval_steps': 500,
        'log_steps': 100,
        'early_stopping_patience': 5
    }
    
    # 实验配置
    EXPERIMENT_CONFIG = {
        'experiment_name': 'transformer_component_analysis',
        'seed': 42,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_gpus': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'mixed_precision': True,
        'save_model': True,
        'save_attention_weights': True,
        'visualization_enabled': True
    }
    
    # 消融实验配置
    ABLATION_CONFIG = {
        'remove_attention': False,
        'remove_positional_encoding': False,
        'remove_feed_forward': False,
        'remove_residual': False,
        'remove_layer_norm': False,
        'remove_dropout': False,
        'single_head_attention': False,
        'no_warmup': False
    }
    
    # 可视化配置
    VISUALIZATION_CONFIG = {
        'attention_heatmap': True,
        'attention_flow': True,
        'gradient_flow': True,
        'loss_curves': True,
        'attention_patterns': True,
        'component_contribution': True,
        'save_format': 'png',
        'dpi': 300,
        'figsize': (12, 8)
    }
    
    # 路径配置
    PATHS = {
        'data_dir': 'data',
        'raw_data_dir': 'data/raw',
        'processed_data_dir': 'data/processed',
        'models_dir': 'results/models',
        'plots_dir': 'results/plots',
        'logs_dir': 'results/logs',
        'report_dir': 'report'
    }
    
    @classmethod
    def get_model_config(cls):
        """获取模型配置"""
        return cls.MODEL_CONFIG
    
    @classmethod
    def get_training_config(cls):
        """获取训练配置"""
        return cls.TRAINING_CONFIG
    
    @classmethod
    def get_data_config(cls):
        """获取数据配置"""
        return cls.DATA_CONFIG
    
    @classmethod
    def get_experiment_config(cls):
        """获取实验配置"""
        return cls.EXPERIMENT_CONFIG
    
    @classmethod
    def get_ablation_config(cls):
        """获取消融实验配置"""
        return cls.ABLATION_CONFIG
    
    @classmethod
    def get_visualization_config(cls):
        """获取可视化配置"""
        return cls.VISUALIZATION_CONFIG
    
    @classmethod
    def get_paths(cls):
        """获取路径配置"""
        return cls.PATHS

# 创建配置实例
config = Config() 