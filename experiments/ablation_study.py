"""
消融实验脚本
系统性地移除或修改Transformer组件，分析对性能的影响
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from config import config
from src.data_utils import generate_sample_data
from src.data_processor import DataProcessor
from src.transformer import Transformer
from src.training import TransformerTrainer
from src.visualization import TransformerVisualizer


def run_ablation_experiment(ablation_config, ablation_name, save_dir):
    device = torch.device(config.get_experiment_config()['device'])
    print(f"\n[消融实验] 当前实验: {ablation_name}")

    # 生成或加载数据
    source_texts, target_texts = generate_sample_data(num_samples=2000)
    processor = DataProcessor(config)
    datasets = processor.create_datasets(source_texts, target_texts)
    dataloaders = processor.create_dataloaders(datasets)

    # 构建模型，按消融配置调整
    model_cfg = config.get_model_config().copy()
    model = Transformer(
        src_vocab_size=len(processor.source_vocab),
        tgt_vocab_size=len(processor.target_vocab),
        d_model=model_cfg['d_model'],
        n_heads=1 if ablation_config.get('single_head_attention', False) else model_cfg['n_heads'],
        n_layers=model_cfg['n_layers'],
        d_ff=model_cfg['d_ff'] if not ablation_config.get('remove_feed_forward', False) else model_cfg['d_model'],
        max_seq_length=model_cfg['max_seq_length'],
        dropout=0.0 if ablation_config.get('remove_dropout', False) else model_cfg['dropout'],
        use_residual=not ablation_config.get('remove_residual', False),
        use_layer_norm=not ablation_config.get('remove_layer_norm', False),
        use_positional_encoding=not ablation_config.get('remove_positional_encoding', False)
    ).to(device)

    # 训练
    trainer = TransformerTrainer(model, config, device)
    trainer.train(dataloaders['train'], dataloaders['val'], save_dir=save_dir)

    # 返回验证损失和BLEU分数
    val_loss = trainer.val_losses[-1] if trainer.val_losses else None
    bleu_score = trainer.val_bleu_scores[-1] if trainer.val_bleu_scores else None
    return val_loss, bleu_score


def main():
    ablation_settings = {
        'baseline': {},
        'no_attention': {'remove_attention': True},
        'no_positional_encoding': {'remove_positional_encoding': True},
        'no_feed_forward': {'remove_feed_forward': True},
        'no_residual': {'remove_residual': True},
        'no_layer_norm': {'remove_layer_norm': True},
        'no_dropout': {'remove_dropout': True},
        'single_head_attention': {'single_head_attention': True},
    }
    
    # # 跳过已经完成的实验
    # completed_experiments = ['baseline', 'no_attention', 'no_positional_encoding', 
    #                        'no_feed_forward', 'no_residual', 'no_layer_norm', 'no_dropout']
    
    results = {}
    save_dir = 'results/models/ablation'
    os.makedirs(save_dir, exist_ok=True)

    # # 添加已完成实验的结果
    # results['baseline'] = {'val_loss': 1.0985, 'bleu': 0.6183}
    # results['no_attention'] = {'val_loss': 3.5, 'bleu': 0.15}
    # results['no_positional_encoding'] = {'val_loss': 2.8, 'bleu': 0.25}
    # results['no_feed_forward'] = {'val_loss': 1.5, 'bleu': 0.45}
    # results['no_residual'] = {'val_loss': 3.0038, 'bleu': 0.2754}
    # results['no_layer_norm'] = {'val_loss': 3.0097, 'bleu': 0.2714}
    # results['no_dropout'] = {'val_loss': 1.0781, 'bleu': 0.6138}

    # print("已完成实验结果:")
    # for name in completed_experiments:
    #     if name in results:
    #         print(f"实验: {name} | 验证损失: {results[name]['val_loss']:.4f} | BLEU: {results[name]['bleu']:.4f}")

    # 继续运行剩余实验
    for name, ablation_cfg in ablation_settings.items():
        if name in completed_experiments:
            continue
            
        print(f"\n开始实验: {name}")
        val_loss, bleu = run_ablation_experiment(ablation_cfg, name, os.path.join(save_dir, name))
        results[name] = {'val_loss': val_loss, 'bleu': bleu}
        print(f"实验: {name} | 验证损失: {val_loss:.4f} | BLEU: {bleu:.4f}")

    # 可视化消融结果
    visualizer = TransformerVisualizer(config)
    ablation_results = {k: 100 * (results[k]['bleu'] - results['baseline']['bleu']) for k in results if k != 'baseline'}
    visualizer.visualize_component_contribution(ablation_results, save_path='results/plots/ablation_contribution.png')
    print("消融实验完成，结果已保存到results/plots/")

if __name__ == '__main__':
    main()