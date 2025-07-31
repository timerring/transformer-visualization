"""
组件分析实验脚本
演示Transformer各组件的作用和可视化
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from config import config
from src.data_utils import generate_sample_data
from src.data_processor import DataProcessor
from src.transformer import Transformer
from src.training import TransformerTrainer
from src.visualization import TransformerVisualizer


def main():
    # 配置
    device = torch.device(config.get_experiment_config()['device'])
    print(f"使用设备: {device}")

    # 生成或加载数据
    print("生成示例数据...")
    source_texts, target_texts = generate_sample_data(num_samples=2000)

    # 数据处理
    print("处理数据...")
    processor = DataProcessor(config)
    datasets = processor.create_datasets(source_texts, target_texts)
    dataloaders = processor.create_dataloaders(datasets)

    # 构建模型
    print("构建Transformer模型...")
    print(f"源词汇表大小: {len(processor.source_vocab)}")
    print(f"目标词汇表大小: {len(processor.target_vocab)}")
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
    print("开始训练...")
    trainer = TransformerTrainer(model, config, device)
    trainer.train(dataloaders['train'], dataloaders['val'], save_dir='results/models')

    # 可视化
    print("可视化分析...")
    visualizer = TransformerVisualizer(config)
    visualizer.visualize_model_architecture(save_path='results/plots/model_architecture.png')
    trainer.plot_training_curves(save_path='results/plots/training_curves.png')

    # 取一批数据做注意力可视化
    batch = next(iter(dataloaders['val']))
    src = batch['source_ids'].transpose(0, 1).to(device)
    tgt = batch['target_ids'].transpose(0, 1).to(device)
    tgt_input = tgt[:-1, :]
    src_mask = trainer.create_padding_mask(batch['source_ids'].to(device))
    tgt_mask = trainer.create_causal_mask(tgt_input.size(0)).to(device)
    model.eval()
    with torch.no_grad():
        output, attn_info = model(src, tgt_input, src_mask, tgt_mask)
    # 可视化注意力
    if attn_info['encoder_attention_weights']:
        visualizer.attention_analyzer.visualize_attention_weights(
            attn_info['encoder_attention_weights'][-1][0].cpu(),
            save_path='results/plots/encoder_attention.png'
        )
    if attn_info['decoder_self_attention_weights']:
        visualizer.attention_analyzer.visualize_attention_weights(
            attn_info['decoder_self_attention_weights'][-1][0].cpu(),
            save_path='results/plots/decoder_self_attention.png'
        )
    if attn_info['decoder_cross_attention_weights']:
        visualizer.attention_analyzer.visualize_attention_weights(
            attn_info['decoder_cross_attention_weights'][-1][0].cpu(),
            save_path='results/plots/decoder_cross_attention.png'
        )
    print("实验完成，结果已保存到results/plots/")

if __name__ == '__main__':
    main()