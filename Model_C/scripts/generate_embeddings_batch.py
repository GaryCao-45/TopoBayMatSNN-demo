#!/usr/bin/env python3
"""
批量生成全局嵌入向量CSV文件的脚本
Generate Global Embeddings CSV File Script - Batch Version

此脚本专门用于提取每个模型对每个材料的全局嵌入向量，
并保存为CSV格式，方便后续的线性验证。

直接在项目根目录运行，避免导入路径问题。
"""

import torch
import numpy as np
import json
import os
import sys
import time
import warnings
from pathlib import Path
import pandas as pd
import argparse
import random

# 设置Python路径并导入模块
import os
import sys

# 确保可以导入src模块
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
src_path = os.path.join(parent_dir, 'src')

if src_path not in sys.path:
    sys.path.insert(0, src_path)

def set_random_seed(seed=42):
    """设置随机种子以确保结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"已设置随机种子: {seed}")

try:
    from src.simplex_data_loader import PerovskiteSimplexDataset
    from src.snn_model import SNN, SNNHamiltonianDynamicsSDE
    print("TEN-FMA核心模块导入成功")
    print(f"从路径导入: {src_path}")
except ImportError as e:
    print("错误: 无法导入必要的模块。")
    print(f"详细错误: {e}")
    print(f"当前Python路径: {sys.path}")
    sys.exit(1)


def load_pretrained_model(model_dir: str, device: torch.device) -> tuple[SNN, dict]:
    """加载预训练模型和其配置"""
    model_path = Path(model_dir) / "pretrained_encoder.pt"
    config_path = Path(model_dir) / "pretrain_args.json"

    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    # 加载配置
    with open(config_path, 'r') as f:
        config = json.load(f)

    # 创建数据集获取特征统计
    dataset = PerovskiteSimplexDataset(data_root="data", load_triangles=True)
    stats = dataset.get_feature_stats()

    # 创建模型
    snn_model = SNN(
        node_input_dim=stats['num_node_features'],
        edge_input_dim=stats['num_edge_features'],
        triangle_input_dim=stats.get('num_triangle_features', 0),
        hidden_dim=config['hidden_dim'],
        output_dim=config['output_dim'],
        num_layers=config['num_layers']
    ).to(device)

    # 加载预训练权重
    checkpoint = torch.load(model_path, map_location=device)

    # 过滤出属于SNN模型的参数（去掉前缀）
    snn_state_dict = {}
    for key, value in checkpoint.items():
        if key.startswith('sde_module.snn_model.'):
            # 移除前缀
            new_key = key.replace('sde_module.snn_model.', '')
            snn_state_dict[new_key] = value

    # 尝试加载，如果有不匹配的参数就跳过
    try:
        snn_model.load_state_dict(snn_state_dict, strict=False)
        print(f"成功加载 {len(snn_state_dict)} 个模型参数")
    except Exception as e:
        print(f"参数加载警告: {e}")
        # 尝试更宽松的加载
        missing_keys, unexpected_keys = snn_model.load_state_dict(snn_state_dict, strict=False)
        if missing_keys:
            print(f"  缺失参数: {len(missing_keys)}")
        if unexpected_keys:
            print(f"  多余参数: {len(unexpected_keys)}")
    snn_model.eval()

    print(f"成功加载模型: {model_dir}")
    return snn_model, config


def run_static_embedding_extraction(model: SNN, data, device: torch.device):
    """直接使用SNN模型提取静态嵌入向量"""
    model.eval()
    model.to(device)
    data = data.to(device)
    with torch.no_grad():
        _, global_embedding = model(data)
    return global_embedding.cpu()


def run_hamiltonian_dynamics(model: SNNHamiltonianDynamicsSDE, data, device: torch.device):
    """运行哈密顿动力学并返回结果"""
    model.eval()
    model.to(device)

    with torch.no_grad():
        q_final, p_final, hamiltonian_history, kinetic_history, potential_history, trajectory_history = model(data)

        # 获取最终的全局嵌入向量
        final_data = data.clone()
        final_data.pos = q_final
        _, global_embedding = model.sde_module.snn_model(final_data)

    return (q_final.cpu(), p_final.cpu(), hamiltonian_history,
            kinetic_history, potential_history, trajectory_history.cpu(), global_embedding.cpu())


def extract_embeddings_for_materials(model_dir: str, materials: list, device: torch.device = None):
    """为指定材料提取全局嵌入向量"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n开始提取模型 {model_dir} 的嵌入向量 (静态模式)")
    print(f"材料列表: {', '.join(materials)}")

    try:
        # 1. 加载预训练模型
        snn_model, config = load_pretrained_model(model_dir, device)

        # 2. 创建数据集
        dataset = PerovskiteSimplexDataset(data_root="data", load_triangles=True)
        if len(dataset) == 0:
            print("✗ 错误: 数据集中没有样本")
            return []

        embeddings_data = []

        for material in materials:
            print(f"处理材料: {material}")

            # 选择指定的材料
            test_sample = None
            for i in range(len(dataset)):
                sample = dataset.get(i)
                if hasattr(sample, 'material_name') and sample.material_name == material:
                    test_sample = sample
                    break

            if test_sample is None:
                print(f"警告: 未找到材料 '{material}'，跳过")
                continue

            data = test_sample.to(device)
            actual_material = getattr(data, 'material_name', 'Unknown')

            # 方案B核心改动：直接调用SNN模型，绕过SDE
            start_time = time.time()
            global_embedding = run_static_embedding_extraction(snn_model, data, device)
            computation_time = time.time() - start_time

            # 移除SDE相关的守恒分数计算
            # hamiltonian_array = np.array(hamiltonian_history)
            # hamiltonian_mean = np.mean(hamiltonian_array)
            # hamiltonian_std = np.std(hamiltonian_array)
            # conservation_score = hamiltonian_std / abs(hamiltonian_mean) if hamiltonian_mean != 0 else float('inf')

            # 6. 保存嵌入向量数据
            embedding_data = {
                'material': actual_material,
                'embedding': global_embedding,
                # 'conservation_score': conservation_score, # 移除
                'computation_time': computation_time
            }

            embeddings_data.append(embedding_data)
            print(f"完成，耗时: {computation_time:.4f}s")

        return embeddings_data

    except Exception as e:
        print(f"嵌入向量提取失败: {e}")
        import traceback
        traceback.print_exc()
        return []


def save_embeddings_to_csv(model_dir: str, embeddings_data: list, output_dir: str = "embeddings"):
    """
    保存全局嵌入向量到CSV文件

    Args:
        model_dir: 模型目录名
        embeddings_data: 嵌入向量数据列表
        output_dir: 输出目录
    """
    Path(output_dir).mkdir(exist_ok=True)

    # 准备数据
    csv_data = []
    embedding_dim = None

    for data in embeddings_data:
        material = data['material']
        embedding = data['embedding'].cpu().numpy().flatten()
        # conservation_score = data['conservation_score'] # 移除
        computation_time = data['computation_time']

        if embedding_dim is None:
            embedding_dim = len(embedding)

        # 创建一行数据：材料名 + 计算时间 + 嵌入向量
        row = {
            'material': material,
            # 'conservation_score': conservation_score, # 移除
            'computation_time': computation_time,
            'model': model_dir
        }

        # 添加嵌入向量的每个维度
        for i, val in enumerate(embedding):
            row[f'emb_{i}'] = val

        csv_data.append(row)

    if not csv_data:
        print("没有嵌入向量数据可保存")
        return None

    # 创建DataFrame并保存
    df = pd.DataFrame(csv_data)
    output_path = Path(output_dir) / f"{model_dir}_embeddings.csv"

    df.to_csv(output_path, index=False, float_format='%.6f')
    print(f"全局嵌入向量已保存至: {output_path}")
    print(f"   数据形状: {len(csv_data)} 行 × {len(df.columns)} 列")
    print(f"   嵌入维度: {embedding_dim}")

    return output_path


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="TEN-FMA全局嵌入向量提取工具")
    parser.add_argument('--model1', type=str, default='model-1-09171648',
                       help='第一个模型目录名')
    parser.add_argument('--model2', type=str, default='model-2-09180905',
                       help='第二个模型目录名')
    parser.add_argument('--model3', type=str, default='model-3-09181323',
                       help='第三个模型目录名')
    parser.add_argument('--model4', type=str, default='model-4-09181820',
                       help='第四个模型目录名')
    parser.add_argument('--materials', type=str, nargs='+',
                       help='要提取嵌入向量的材料名称')
    parser.add_argument('--all-materials', action='store_true',
                       help='提取所有5个材料的嵌入向量 (CsPbCl3, CsPbBr3, CH3NH3GeI3, CH3NH3PbI3, CsPbI3)')
    parser.add_argument('--output-dir', type=str, default='embeddings',
                       help='输出目录 (相对于模型目录)')
    parser.add_argument('--device', type=str, default=None,
                       help='计算设备 (cuda/cpu/auto)')

    args = parser.parse_args()

    # 设置随机种子
    set_random_seed(42)

    print("TEN-FMA全局嵌入向量提取工具")
    print("="*60)

    # 设置设备
    if args.device == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif args.device == 'cpu':
        device = torch.device('cpu')
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"使用设备: {device}")
    if torch.cuda.is_available() and device.type == 'cuda':
        print(f"CUDA设备: {torch.cuda.get_device_name(device)}")

    # 确定材料列表
    if args.all_materials:
        materials = ['CsPbCl3', 'CsPbBr3', 'CH3NH3GeI3', 'CH3NH3PbI3', 'CsPbI3']
        print(f"处理所有5个材料: {', '.join(materials)}")
    elif args.materials:
        materials = args.materials
        print(f"处理指定材料: {', '.join(materials)}")
    else:
        print("错误: 请使用 --materials 指定材料或使用 --all-materials 处理所有材料")
        parser.print_help()
        sys.exit(1)

    # 处理每个模型
    all_embeddings_data = []

    for model_dir in [args.model1, args.model2, args.model3, args.model4]:
        if not Path(model_dir).exists():
            print(f"警告: 模型目录不存在: {model_dir}")
            continue

        # 提取嵌入向量
        embeddings_data = extract_embeddings_for_materials(model_dir, materials, device)

        if embeddings_data:
            # 保存到CSV
            model_short_name = Path(model_dir).name
            embeddings_dir = Path(model_dir) / args.output_dir
            csv_path = save_embeddings_to_csv(model_short_name, embeddings_data, str(embeddings_dir))

            all_embeddings_data.extend(embeddings_data)

            print(f"模型 {model_short_name} 处理完成")
        else:
            print(f"模型 {model_dir} 处理失败")

    # 输出最终统计
    print(f"\n{'='*60}")
    print("嵌入向量提取完成总结:")
    print(f"{'='*60}")

    if all_embeddings_data:
        total_materials = len(set([d['material'] for d in all_embeddings_data]))
        total_models = len(set([Path(d.get('model', '')).name for d in all_embeddings_data if d.get('model')]))

        print(f"成功提取: {len(all_embeddings_data)} 个嵌入向量")
        print(f"   涵盖材料: {total_materials} 个")
        print(f"   涵盖模型: {total_models} 个")

        # 移除守恒分数统计
        # print("\n守恒分数统计:")
        # material_scores = {}
        # for data in all_embeddings_data:
        #     material = data['material']
        #     score = data['conservation_score']
        #     if material not in material_scores:
        #         material_scores[material] = []
        #     material_scores[material].append(score)
        #
        # for material, scores in material_scores.items():
        #     avg_score = np.mean(scores)
        #     min_score = np.min(scores)
        #     print(f"   - {material:<15}: 平均守恒分数={avg_score:.6f} (越小越好), 最小={min_score:.6f}")

        print(f"\nCSV文件保存在各个模型目录的 {args.output_dir}/ 文件夹中")
        print("现在可以使用这些CSV文件进行线性验证！")
    else:
        print("未提取到任何嵌入向量")


if __name__ == "__main__":
    main()
