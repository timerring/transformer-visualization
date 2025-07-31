"""
注意力可视化实验脚本
系统性展示不同层和不同头的注意力分布与演化
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from config import config
from src.data_utils import generate_sample_data
from src.data_processor import DataProcessor
from src.transformer import Transformer
from src.training import TransformerTrainer
from src.visualization import TransformerVisualizer


def main():
    device = torch.device(config.get_experiment_config()['device'])
    print(f"使用设备: {device}")

    # 生成或加载数据
    print("生成示例数据...")
    source_texts, target_texts = generate_sample_data(num_samples=1000)
    processor = DataProcessor(config)
    datasets = processor.create_datasets(source_texts, target_texts)
    dataloaders = processor.create_dataloaders(datasets)

    # 构建模型
    print("构建Transformer模型...")
    model_cfg = config.get_model_config()
    model = Transformer(
        src_vocab_size=len(processor.source_vocab),
        tgt_vocab_size=len(processor.target_vocab),
        d_model=model_cfg['d_model'],
        n_heads=model_cfg['n_heads'],
        n_layers=model_cfg['n_layers'],
        d_ff=model_cfg['d_ff'],
        max_seq_length=model_cfg['max_seq_length'],
        dropout=model_cfg['dropout'],
        use_residual=True,
        use_layer_norm=True,
        use_positional_encoding=True
    ).to(device)

    # 训练
    print("训练模型...")
    trainer = TransformerTrainer(model, config, device)
    trainer.train(dataloaders['train'], dataloaders['val'], save_dir='results/models/attention_vis')

    # 可视化
    print("注意力可视化分析...")
    visualizer = TransformerVisualizer(config)
    batch = next(iter(dataloaders['val']))
    src = batch['source_ids'].transpose(0, 1).to(device)
    tgt = batch['target_ids'].transpose(0, 1).to(device)
    tgt_input = tgt[:-1, :]
    src_mask = trainer.create_padding_mask(batch['source_ids'].to(device))
    tgt_mask = trainer.create_causal_mask(tgt_input.size(0)).to(device)
    model.eval()
    with torch.no_grad():
        output, attn_info = model(src, tgt_input, src_mask, tgt_mask)
    # 展示不同层的注意力分布
    if attn_info['encoder_attention_weights']:
        visualizer.visualize_attention_evolution(attn_info['encoder_attention_weights'], save_path='results/plots/encoder_attention_evolution.png')
    if attn_info['decoder_self_attention_weights']:
        visualizer.visualize_attention_evolution(attn_info['decoder_self_attention_weights'], save_path='results/plots/decoder_self_attention_evolution.png')
    if attn_info['decoder_cross_attention_weights']:
        visualizer.visualize_attention_evolution(attn_info['decoder_cross_attention_weights'], save_path='results/plots/decoder_cross_attention_evolution.png')
    print("注意力可视化实验完成，结果已保存到results/plots/")

if __name__ == '__main__':
    main()