"""
预训练模型可视化脚本：TEN-FMA哈密顿SDE动力学可视化
Pretrained Model Visualization Script: TEN-FMA Hamiltonian SDE Dynamics Visualization

本脚本专门用于可视化两个预训练模型：
1. model-1-09171648/: 第一个预训练模型
2. model-2-09180905/: 第二个预训练模型

功能包括：
- 哈密顿量能量守恒可视化（检查是否为直线）
- 三维原子轨迹交互式可视化
- 自动保存可视化结果到对应模型目录

使用方法：
python visualize_pretrained_models.py --model1 model-1-09171648 --model2 model-2-09180905
"""

import torch
import numpy as np
import json
import os
import sys
import time
import warnings
from pathlib import Path
from pymatgen.core import Structure
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import argparse
import pandas as pd

# 设置Python路径并导入模块
import os
import sys

# 确保可以导入src模块
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
src_path = os.path.join(parent_dir, 'src')

if src_path not in sys.path:
    sys.path.insert(0, src_path)

try:
    from src.simplex_data_loader import PerovskiteSimplexDataset
    from src.snn_model import SNN, SNNHamiltonianDynamicsSDE
    print("TEN-FMA核心模块导入成功")
    print(f" 从路径导入: {src_path}")
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


def save_embeddings_to_csv(model_dir: str, embeddings_data: list, output_dir: str = "embeddings"):
    """
    保存全局嵌入向量到CSV文件

    Args:
        model_dir: 模型目录名
        embeddings_data: 嵌入向量数据列表，每个元素为字典
                        {'material': str, 'embedding': torch.Tensor, 'conservation_score': float}
        output_dir: 输出目录
    """
    Path(output_dir).mkdir(exist_ok=True)

    # 准备数据
    csv_data = []
    embedding_dim = None

    for data in embeddings_data:
        material = data['material']
        embedding = data['embedding'].cpu().numpy().flatten()
        conservation_score = data['conservation_score']

        if embedding_dim is None:
            embedding_dim = len(embedding)

        # 创建一行数据：材料名 + 守恒分数 + 嵌入向量
        row = {
            'material': material,
            'conservation_score': conservation_score,
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


def plot_hamiltonian_conservation(hamiltonian_history: list, kinetic_history: list,
                                potential_history: list, material_name: str,
                                model_name: str, output_dir: str = "plots"):
    """绘制美观的能量守恒图像：哈密顿量、动能、势能"""
    Path(output_dir).mkdir(exist_ok=True)

    # 设置美观的图表样式
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

    steps = range(len(hamiltonian_history))

    # 专业的科学配色方案
    colors = {
        'hamiltonian': '#2E86AB',  # 深蓝色 - 哈密顿量
        'kinetic': '#A23B72',      # 深紫红色 - 动能
        'potential': '#F18F01'     # 橙色 - 势能
    }

    # 左图：完整能量演化
    ax1.plot(steps, hamiltonian_history,
            color=colors['hamiltonian'], linewidth=2.5,
            marker='o', markersize=4, markerfacecolor='white', markeredgewidth=1.5,
            label=f'Hamiltonian (H = T + V)', alpha=0.9)

    ax1.plot(steps, kinetic_history,
            color=colors['kinetic'], linewidth=2.0,
            marker='s', markersize=3, markerfacecolor='white', markeredgewidth=1.2,
            label=f'Kinetic Energy (T)', alpha=0.8)

    ax1.plot(steps, potential_history,
            color=colors['potential'], linewidth=2.0,
            marker='^', markersize=3, markerfacecolor='white', markeredgewidth=1.2,
            label=f'Potential Energy (V)', alpha=0.8)

    ax1.set_xlabel('Integration Step', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Energy (Arbitrary Units)', fontsize=14, fontweight='bold')
    ax1.set_title(f'Energy Evolution\n{material_name} - {model_name}', fontsize=16, fontweight='bold', pad=20)
    ax1.legend(loc='upper right', fontsize=12, frameon=True, fancybox=True, shadow=True, framealpha=0.9)
    ax1.grid(True, linestyle='--', alpha=0.7, linewidth=0.8)

    # 右图：哈密顿量守恒性分析（重点检查是否为直线）
    hamiltonian_array = np.array(hamiltonian_history)
    hamiltonian_mean = np.mean(hamiltonian_array)
    hamiltonian_std = np.std(hamiltonian_array)
    conservation_score = hamiltonian_std / abs(hamiltonian_mean) if hamiltonian_mean != 0 else float('inf')

    ax2.plot(steps, hamiltonian_history,
            color=colors['hamiltonian'], linewidth=3.0,
            marker='o', markersize=5, markerfacecolor='white', markeredgewidth=2.0,
            label='.3f', alpha=0.9)

    # 添加平均值参考线
    ax2.axhline(y=hamiltonian_mean, color='red', linestyle='--', linewidth=2,
               label='.3f', alpha=0.8)

    ax2.fill_between(steps,
                    hamiltonian_mean - hamiltonian_std,
                    hamiltonian_mean + hamiltonian_std,
                    color='red', alpha=0.1, label='.3f')

    ax2.set_xlabel('Integration Step', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Hamiltonian (H)', fontsize=14, fontweight='bold')
    ax2.set_title(f'Hamiltonian Conservation Analysis\nConservation Score: {conservation_score:.6f}',
                 fontsize=16, fontweight='bold', pad=20)
    ax2.legend(loc='upper right', fontsize=12, frameon=True, fancybox=True, shadow=True, framealpha=0.9)
    ax2.grid(True, linestyle='--', alpha=0.7, linewidth=0.8)

    # 设置统一的网格和边框样式
    for ax in [ax1, ax2]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)
        ax.tick_params(axis='both', which='major', labelsize=11)

    plt.tight_layout()
    output_path = Path(output_dir) / f"hamiltonian_conservation_{material_name}_{model_name}.png"
    plt.savefig(output_path, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

    print(f"美观的能量守恒图像已保存至: {output_path}")
    print(".6f")
    return conservation_score


def plot_atomic_trajectory(trajectory: torch.Tensor, structure: Structure,
                         material_name: str, model_name: str, output_dir: str = "plots"):
    """生成美观的三维可交互原子轨迹HTML文件"""
    Path(output_dir).mkdir(exist_ok=True)

    # 关键修复：在绘图前，将张量从计算图中分离
    trajectory_vis = trajectory.detach().cpu()

    # 专业的元素配色方案（更鲜艳和对比度更高）
    element_colors = {
        'Pb': '#2C3E50',   # 深蓝灰色 - 铅
        'I': '#9B59B6',    # 紫色 - 碘
        'Cs': '#27AE60',   # 绿色 - 铯
        'C': '#34495E',    # 深灰色 - 碳
        'N': '#3498DB',    # 蓝色 - 氮
        'H': '#95A5A6',    # 浅灰色 - 氢
        'Ge': '#F39C12',   # 橙色 - 锗
        'Cl': '#1ABC9C',   # 青绿色 - 氯
        'Br': '#E67E22',   # 橙红色 - 溴
        'Sn': '#8E44AD'    # 深紫色 - 锡
    }

    # 原子大小映射（基于原子半径）
    element_sizes = {
        'Pb': 12, 'I': 10, 'Cs': 14, 'C': 6,
        'N': 5, 'H': 3, 'Ge': 8, 'Cl': 7, 'Br': 9, 'Sn': 10
    }

    atom_colors = [element_colors.get(site.specie.symbol, '#7F8C8D') for site in structure.sites]
    atom_symbols = [site.specie.symbol for site in structure.sites]
    atom_sizes = [element_sizes.get(symbol, 8) for symbol in atom_symbols]

    fig_data = []
    num_steps, num_atoms, _ = trajectory_vis.shape

    # 计算轨迹范围用于更好的视角设置
    all_positions = trajectory_vis.view(-1, 3)
    x_range = all_positions[:, 0].max() - all_positions[:, 0].min()
    y_range = all_positions[:, 1].max() - all_positions[:, 1].min()
    z_range = all_positions[:, 2].max() - all_positions[:, 2].min()
    max_range = max(x_range, y_range, z_range)

    # 绘制轨迹线（更细致的样式）
    for i in range(num_atoms):
        fig_data.append(go.Scatter3d(
            x=trajectory_vis[:, i, 0], y=trajectory_vis[:, i, 1], z=trajectory_vis[:, i, 2],
            mode='lines',
            line=dict(color=atom_colors[i], width=3, dash='solid'),
            opacity=0.7,
            name=f'{atom_symbols[i]}{i+1} Trajectory',
            showlegend=i < 5  # 只显示前5个原子的图例，避免过于拥挤
        ))

    # 绘制初始位置原子（更大更明显）
    fig_data.append(go.Scatter3d(
        x=trajectory_vis[0, :, 0], y=trajectory_vis[0, :, 1], z=trajectory_vis[0, :, 2],
        mode='markers',
        marker=dict(
            color=atom_colors,
            size=[s*1.5 for s in atom_sizes],
            symbol='circle',
            line=dict(color='white', width=2),
            opacity=0.9
        ),
        name='Initial Positions',
        text=[f'{symbol} (Initial)' for symbol in atom_symbols],
        hovertemplate='<b>%{text}</b><br>Position: (%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>'
    ))

    # 绘制最终位置原子（不同形状以示区别）
    fig_data.append(go.Scatter3d(
        x=trajectory_vis[-1, :, 0], y=trajectory_vis[-1, :, 1], z=trajectory_vis[-1, :, 2],
        mode='markers',
        marker=dict(
            color=atom_colors,
            size=[s*1.8 for s in atom_sizes],
            symbol='diamond',
            line=dict(color='black', width=2),
            opacity=0.95
        ),
        name='Final Positions',
        text=[f'{symbol} (Final)' for symbol in atom_symbols],
        hovertemplate='<b>%{text}</b><br>Position: (%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>'
    ))

    fig = go.Figure(data=fig_data)

    # 美化布局
    fig.update_layout(
        title=dict(
            text=f'<b>3D Atomic Trajectory in Hamiltonian SDE Dynamics</b><br><sub>{material_name} - {model_name}</sub>',
            x=0.5,
            font=dict(size=18, family="Arial, sans-serif")
        ),
        scene=dict(
            xaxis_title=dict(text='X Position (Å)', font=dict(size=14)),
            yaxis_title=dict(text='Y Position (Å)', font=dict(size=14)),
            zaxis_title=dict(text='Z Position (Å)', font=dict(size=14)),
            bgcolor='rgba(240,240,240,0.9)',
            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.8)', gridwidth=1,
                      range=[trajectory_vis[:,:,0].min()-max_range*0.1, trajectory_vis[:,:,0].max()+max_range*0.1]),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.8)', gridwidth=1,
                      range=[trajectory_vis[:,:,1].min()-max_range*0.1, trajectory_vis[:,:,1].max()+max_range*0.1]),
            zaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.8)', gridwidth=1,
                      range=[trajectory_vis[:,:,2].min()-max_range*0.1, trajectory_vis[:,:,2].max()+max_range*0.1]),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5),
                center=dict(x=0, y=0, z=0)
            ),
            aspectmode='cube'  # 保持立方体比例
        ),
        width=1200,
        height=800,
        margin=dict(l=0, r=0, b=0, t=60),
        legend=dict(
            x=0.02, y=0.98,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='rgba(0,0,0,0.2)',
            borderwidth=1,
            font=dict(size=10)
        ),
        font=dict(family="Arial, sans-serif", size=12),
        paper_bgcolor='white',
        plot_bgcolor='white'
    )

    output_path = Path(output_dir) / f"atomic_trajectory_{material_name}_{model_name}.html"
    fig.write_html(output_path)
    print(f"美观的原子轨迹交互式可视化HTML已保存至: {output_path}")

    return output_path


def visualize_model(model_dir: str, material_name: str = "CsPbI3", device: torch.device = None,
                   save_embeddings: bool = True):
    """可视化单个模型"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"开始可视化模型: {model_dir}")
    print(f"{'='*60}")

    try:
        # 1. 加载预训练模型
        snn_model, config = load_pretrained_model(model_dir, device)

        # 2. 创建数据集并选择测试样本
        dataset = PerovskiteSimplexDataset(data_root="data", load_triangles=True)
        if len(dataset) == 0:
            print("错误: 数据集中没有样本")
            return None

        # 选择指定的材料或第一个可用样本
        test_sample = None
        for i in range(len(dataset)):
            sample = dataset.get(i)
            if hasattr(sample, 'material_name') and sample.material_name == material_name:
                test_sample = sample
                break

        if test_sample is None:
            print(f"警告: 未找到材料 '{material_name}'，使用第一个可用样本")
            test_sample = dataset.get(0)

        data = test_sample.to(device)
        actual_material = getattr(data, 'material_name', 'Unknown')

        # 3. 创建SDE动力学模型
        sde_dynamics_model = SNNHamiltonianDynamicsSDE(
            snn_model=snn_model,
            integration_steps=config.get('sde_steps', 10),
            integration_time=1.0,
            temperature=config.get('temperature', 0.05),
            friction=config.get('friction', 0.01)
        )

        # 4. 运行哈密顿动力学
        print(f"正在运行哈密顿动力学模拟 ({actual_material})...")
        start_time = time.time()
        results = run_hamiltonian_dynamics(sde_dynamics_model, data, device)
        q_final, p_final, hamiltonian_history, kinetic_history, potential_history, trajectory_history, global_embedding = results
        end_time = time.time()

        # 5. 生成可视化
        model_short_name = Path(model_dir).name
        plots_dir = Path(model_dir) / "visualizations"
        plots_dir.mkdir(exist_ok=True)

        # 哈密顿量守恒分析
        conservation_score = plot_hamiltonian_conservation(
            hamiltonian_history, kinetic_history, potential_history,
            actual_material, model_short_name, str(plots_dir)
        )

        # 加载结构文件用于3D可视化
        cif_path = Path("data") / f"{actual_material}-gpaw-optimized.cif"
        if cif_path.exists():
            structure = Structure.from_file(cif_path)
            trajectory_path = plot_atomic_trajectory(
                trajectory_history, structure, actual_material, model_short_name, str(plots_dir)
            )
        else:
            print(f"警告: CIF文件不存在: {cif_path}")
            trajectory_path = None

        # 6. 输出结果总结
        print(f"\n模型 {model_short_name} 可视化结果:")
        print(f"   材料: {actual_material}")
        print(f"   轨迹步数: {len(hamiltonian_history)}")
        print(f"   可视化文件保存在: {plots_dir}")

        # 7. 保存嵌入向量数据
        embedding_data = None
        if save_embeddings:
            embedding_data = {
                'material': actual_material,
                'embedding': global_embedding,
                'conservation_score': conservation_score
            }

        return {
            'model_name': model_short_name,
            'material': actual_material,
            'conservation_score': conservation_score,
            'computation_time': end_time - start_time,
            'plots_dir': plots_dir,
            'trajectory_path': trajectory_path,
            'embedding_data': embedding_data
        }

    except Exception as e:
        print(f"可视化失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def batch_visualize_materials(model_dir: str, materials: list, device: torch.device = None,
                           save_embeddings: bool = True):
    """批量可视化多个材料"""
    print(f"\n{'='*70}")
    print(f"开始批量可视化模型: {model_dir}")
    print(f"材料列表: {', '.join(materials)}")
    print(f"{'='*70}")

    results = []
    embeddings_data = []

    for material in materials:
        print(f"\n处理材料: {material}")
        print("-" * 50)
        result = visualize_model(model_dir, material, device, save_embeddings)
        if result:
            results.append(result)
            if save_embeddings and result.get('embedding_data'):
                embeddings_data.append(result['embedding_data'])

    # 保存嵌入向量到CSV
    if save_embeddings and embeddings_data:
        model_short_name = Path(model_dir).name
        embeddings_dir = Path(model_dir) / "embeddings"
        save_embeddings_to_csv(model_short_name, embeddings_data, str(embeddings_dir))

    # 输出批量处理总结
    if results:
        print(f"\n{'='*70}")
        print(f"批量可视化完成总结 ({model_dir}):")
        print(f"{'='*70}")

        for result in results:
            print(f"{result['material']}: 守恒分数 = {result['conservation_score']:.6f}")

        # 找出每个材料的最佳模型守恒性
        best_scores = {r['material']: r['conservation_score'] for r in results}
        best_material = min(best_scores, key=best_scores.get)
        print(f"\n最佳守恒性材料: {best_material} (分数: {best_scores[best_material]:.6f})")

    return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="TEN-FMA预训练模型可视化工具")
    parser.add_argument('--model1', type=str, default='model-1-09171648',
                       help='第一个模型目录名')
    parser.add_argument('--model2', type=str, default='model-2-09180905',
                       help='第二个模型目录名')
    parser.add_argument('--model3', type=str, default='model-3-09181323',
                       help='第三个模型目录名')
    parser.add_argument('--model4', type=str, default='model-4-09181820',
                          help='第四个模型目录名')
    parser.add_argument('--material', type=str, default='CsPbI3',
                       help='要可视化的材料名称')
    parser.add_argument('--materials', type=str, nargs='+',
                       help='要可视化的多个材料名称 (空格分隔)')
    parser.add_argument('--batch-all', action='store_true',
                       help='批量可视化所有5个材料 (CsPbCl3, CsPbBr3, CH3NH3GeI3, CH3NH3PbI3, CsPbI3)')
    parser.add_argument('--device', type=str, default=None,
                       help='计算设备 (cuda/cpu/auto)')
    parser.add_argument('--save-embeddings', action='store_true', default=True,
                       help='保存全局嵌入向量到CSV文件 (默认开启)')
    parser.add_argument('--no-save-embeddings', action='store_true',
                       help='不保存全局嵌入向量到CSV文件')

    args = parser.parse_args()

    print("TEN-FMA预训练模型可视化工具")
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

    # 确定是否保存嵌入向量
    save_embeddings = args.save_embeddings and not args.no_save_embeddings
    if save_embeddings:
        print("嵌入向量保存模式: 开启 (将保存到CSV文件)")
    else:
        print("嵌入向量保存模式: 关闭")

    # 确定要可视化的材料列表
    if args.batch_all:
        # 批量可视化所有5个材料
        all_materials = ['CsPbCl3', 'CsPbBr3', 'CH3NH3GeI3', 'CH3NH3PbI3', 'CsPbI3']
        print(f"批量可视化模式: 处理所有 {len(all_materials)} 个材料")
        print(f"   材料列表: {', '.join(all_materials)}")
    elif args.materials:
        # 指定多个材料
        all_materials = args.materials
        print(f"多材料可视化模式: 处理 {len(all_materials)} 个指定材料")
        print(f"   材料列表: {', '.join(all_materials)}")
    else:
        # 单个材料模式
        all_materials = [args.material]
        print(f"单材料可视化模式: {args.material}")

    # 可视化所有模型和材料组合
    all_results = []

    model_dirs = [args.model1, args.model2, args.model3]
    if hasattr(args, 'model4') and args.model4:
        model_dirs.append(args.model4)

    for model_dir in model_dirs:
        if not Path(model_dir).exists():
            print(f"警告: 模型目录不存在: {model_dir}")
            continue

        if len(all_materials) > 1:
            # 批量可视化多个材料
            model_results = batch_visualize_materials(model_dir, all_materials, device, save_embeddings)
            all_results.extend(model_results)
        else:
            # 单个材料可视化
            result = visualize_model(model_dir, all_materials[0], device, save_embeddings)
            if result:
                all_results.append(result)

    # 输出最终对比总结
    if len(all_results) >= 2:
        print(f"\n{'='*80}")
        print("最终对比总结 (所有模型和材料):")
        print(f"{'='*80}")

        # 按模型分组显示结果
        model_groups = {}
        for result in all_results:
            model_name = result['model_name']
            if model_name not in model_groups:
                model_groups[model_name] = []
            model_groups[model_name].append(result)

        for model_name, results in model_groups.items():
            print(f"\n模型: {model_name}")
            print(f"   处理了 {len(results)} 个材料")

            for result in results:
                print(".6f")

            # 计算该模型的平均守恒分数
            avg_score = np.mean([r['conservation_score'] for r in results])
            print(".6f")

        # 找出全局最佳
        all_scores = [(r['model_name'], r['material'], r['conservation_score']) for r in all_results]
        best_model, best_material, best_score = min(all_scores, key=lambda x: x[2])

        print(f"\n全局最佳组合:")
        print(f"   模型: {best_model}")
        print(f"   材料: {best_material}")
        print(".6f")

    print(f"\n可视化任务完成！共处理了 {len(all_results)} 个模型-材料组合")

    if save_embeddings:
        print("全局嵌入向量CSV文件已保存到各个模型目录的embeddings文件夹")
        print("   格式: 材料名 + 守恒分数 + 嵌入向量维度")

    print("查看各个模型目录下的:")
    print("   • visualizations/ 文件夹 (PNG和HTML可视化文件)")
    if save_embeddings:
        print("   • embeddings/ 文件夹 (CSV嵌入向量文件)")


if __name__ == "__main__":
    main()
