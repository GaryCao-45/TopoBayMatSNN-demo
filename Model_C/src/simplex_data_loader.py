"""
增强数据加载模块：支持0-1-2单纯形的完整拓扑特征
Enhanced Data Loading Module: Support for full topological features of 0-1-2 simplices

Demo version: Framework retained, core loading logic simplified.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset
from typing import List, Tuple, Optional, Dict, Any
import warnings
from pathlib import Path
from collections import defaultdict
from pymatgen.core import Structure

class SimplexComplexData(Data):
    """
    扩展的数据对象，支持单纯形复合体的完整拓扑信息
    Extended data object supporting complete topological information of simplicial complexes
    
    Demo: Basic structure retained.
    """
    
    def __init__(self, 
                 x=None,              # 节点特征 (0-单纯形)
                 edge_index=None,     # 边索引 (1-单纯形连接)
                 edge_attr=None,      # 边特征 (1-单纯形)
                 pos=None,            # 节点坐标
                 triangle_index=None, # 三角形索引 (2-单纯形连接)
                 triangle_attr=None,  # 三角形特征 (2-单纯形)
                 **kwargs):
        """
        初始化单纯形复合体数据对象
        
        Args:
            x: 节点特征 [num_nodes, node_features]
            edge_index: 边索引 [2, num_edges] 
            edge_attr: 边特征 [num_edges, edge_features]
            pos: 节点坐标 [num_nodes, 3]
            triangle_index: 三角形索引 [3, num_triangles]
            triangle_attr: 三角形特征 [num_triangles, triangle_features]
        """
        super().__init__(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos, **kwargs)
        
        # 显式地将face属性设为None，以避免PyG的弃用警告
        self.face = None

        # 2-单纯形信息
        self.triangle_index = triangle_index
        self.triangle_attr = triangle_attr
        
        # 单纯形统计信息
        if triangle_index is not None:
            self.num_triangles = triangle_index.shape[1]
        else:
            self.num_triangles = 0
    
    def __inc__(self, key: str, value: Any, *args, **kwargs):
        """增量索引处理，用于批处理"""
        if key == 'triangle_index':
            return self.x.size(0)  # 三角形索引需要按节点数增加
        else:
            return super().__inc__(key, value, *args, **kwargs)
    
    def __cat_dim__(self, key: str, value: Any, *args, **kwargs):
        """连接维度处理，用于批处理"""
        if key == 'triangle_index':
            return 1  # 沿第二维连接（num_triangles维度）
        elif key == 'triangle_attr':
            return 0  # 沿第一维连接（features维度）
        else:
            return super().__cat_dim__(key, value, *args, **kwargs)
    
    def to(self, device, *args, **kwargs):
        """覆盖to方法以确保所有自定义属性也被移动到指定设备"""
        # 调用父类的to方法
        data = super().to(device, *args, **kwargs)
        
        # 确保自定义属性也被移动
        if hasattr(data, 'triangle_index') and data.triangle_index is not None:
            data.triangle_index = data.triangle_index.to(device)
        
        if hasattr(data, 'triangle_attr') and data.triangle_attr is not None:
            data.triangle_attr = data.triangle_attr.to(device)
            
        if hasattr(data, 'pos') and data.pos is not None:
            data.pos = data.pos.to(device)
        
        return data

class PerovskiteSimplexDataset(Dataset):
    """
    钙钛矿材料单纯形复合体数据集类
    Perovskite materials simplicial complex dataset class
    
    整合0-1-2单纯形的完整拓扑特征
    Integrates complete topological features of 0-1-2 simplices
    
    Demo: Loading framework retained, actual data loading placeholder.
    """
    
    def __init__(self, 
                 data_root: str = "data",
                 transform=None, 
                 pre_transform=None,
                 normalize_features: bool = True,
                 load_triangles: bool = True,
                 enable_augmentation: bool = False,
                 augmentation_noise_std: float = 0.05,
                 include_materials: Optional[List[str]] = None):
        """
        初始化单纯形数据集
        
        Args:
            data_root: 数据文件夹路径
            transform: 动态变换
            pre_transform: 预处理变换 
            normalize_features: 是否标准化特征
            load_triangles: 是否加载2-单纯形特征
            enable_augmentation: 是否启用几何数据增强（用于CGP对比学习）
            augmentation_noise_std: 几何噪声的标准差
            include_materials: (可选) 指定要加载的材料名称列表
        """
        self.data_root = Path(data_root)
        self.normalize_features = normalize_features
        self.load_triangles = load_triangles
        self.enable_augmentation = enable_augmentation
        self.augmentation_noise_std = augmentation_noise_std
        
        # 获取所有材料的基础名称
        self.material_names = self._get_material_names()
        print(f"在 {data_root} 中发现 {len(self.material_names)} 个材料: {self.material_names}")

        # 如果指定了要包含的材料，则进行筛选
        if include_materials is not None:
            self.material_names = [name for name in self.material_names if name in include_materials]
            print(f"筛选后，将加载 {len(self.material_names)} 个指定材料: {self.material_names}")
        
        # 预处理数据
        self.processed_data = []
        self._process_all_materials()
        
        super(PerovskiteSimplexDataset, self).__init__(data_root, transform, pre_transform)
    
    def _get_material_names(self) -> List[str]:
        """获取所有材料的基础名称"""
        # Demo: Placeholder for material names
        return ["CH3NH3GeI3", "CH3NH3PbI3", "CsPbBr3", "CsPbCl3", "CsPbI3"]
    
    @property
    def processed_file_names(self):
        """A list of files in the processed_dir which are already processed."""
        return [f'data_{material}.pt' for material in self.material_names]

    def _load_simplex_features(self, material_name: str) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
        """
        加载0-1-2单纯形特征数据
        
        Args:
            material_name: 材料名称 (e.g., "CH3NH3PbI3")
            
        Returns:
            (0-simplex_df, 1-simplex_df, 2-simplex_df)
        """
        # Demo: Placeholder - simulate loading with dummy data
        # In full version, load from CSV files
        num_atoms = 20  # Placeholder
        num_edges = 60  # Placeholder
        num_triangles = 100  # Placeholder if load_triangles
        
        simplex_0_df = pd.DataFrame(np.random.rand(num_atoms, 40), columns=[f'node_feat_{i}' for i in range(40)])
        simplex_1_df = pd.DataFrame(np.random.rand(num_edges, 20), columns=[f'edge_feat_{i}' for i in range(20)])
        simplex_2_df = pd.DataFrame(np.random.rand(num_triangles, 15), columns=[f'triangle_feat_{i}' for i in range(15)]) if self.load_triangles else None
        
        print(f"材料 {material_name}:")
        print(f"  - 0-单纯形特征: {simplex_0_df.shape}")
        print(f"  - 1-单纯形特征: {simplex_1_df.shape}")
        if simplex_2_df is not None:
            print(f"  - 2-单纯形特征: {simplex_2_df.shape}")
        else:
            print(f"  - 2-单纯形特征: 跳过加载")
        
        return simplex_0_df, simplex_1_df, simplex_2_df
    
    def _normalize_features_if_needed(self, features: torch.Tensor) -> torch.Tensor:
        """标准化特征（可选）"""
        if not self.normalize_features:
            return features
        
        # 避免除零错误
        std = features.std(dim=0)
        std = torch.where(std == 0, torch.ones_like(std), std)
        
        normalized = (features - features.mean(dim=0)) / std
        
        # 处理NaN值
        normalized = torch.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return normalized
    
    def _extract_triangle_connections(self, simplex_2_df: pd.DataFrame) -> torch.Tensor:
        """
        从2-单纯形数据中提取三角形连接信息
        
        Args:
            simplex_2_df: 2-单纯形特征DataFrame
            
        Returns:
            triangle_index: [3, num_triangles] 三角形索引张量
        """
        if simplex_2_df is None or len(simplex_2_df) == 0:
            return torch.empty((3, 0), dtype=torch.long)
        
        # Demo: Placeholder triangle connections
        num_triangles = len(simplex_2_df)
        triangle_index = torch.randint(0, 20, (3, num_triangles), dtype=torch.long)  # Dummy indices
        
        return triangle_index
    
    def _extract_triangle_features(self, simplex_2_df: pd.DataFrame) -> torch.Tensor:
        """
        从2-单纯形数据中提取特征向量
        
        Args:
            simplex_2_df: 2-单纯形特征DataFrame
            
        Returns:
            triangle_features: [num_triangles, triangle_features] 特征张量
        """
        if simplex_2_df is None or len(simplex_2_df) == 0:
            return torch.empty((0, 0), dtype=torch.float)
        
        # Demo: Use all columns as features (placeholder)
        triangle_features = torch.tensor(simplex_2_df.values, dtype=torch.float)
        triangle_features = self._normalize_features_if_needed(triangle_features)
        
        return triangle_features
    
    def _create_simplex_data(self, material_name: str) -> SimplexComplexData:
        """
        为单个材料创建单纯形复合体数据对象
        
        Args:
            material_name: 材料名称
            
        Returns:
            SimplexComplexData对象
        """
        # 加载单纯形特征
        simplex_0_df, simplex_1_df, simplex_2_df = self._load_simplex_features(material_name)
        
        # (新增) 加载结构文件以获取坐标
        # Demo: Placeholder positions
        positions = torch.rand(20, 3)  # Dummy positions for 20 atoms
        
        # 提取节点特征（0-单纯形）
        node_features = torch.tensor(simplex_0_df.values, dtype=torch.float).cpu()
        node_features = self._normalize_features_if_needed(node_features)
        
        # 提取边索引和边特征（1-单纯形）
        # Demo: Dummy edge index and attr
        edge_index = torch.randint(0, 20, (2, 60), dtype=torch.long).cpu()
        edge_features = torch.tensor(simplex_1_df.values, dtype=torch.float).cpu()
        edge_features = self._normalize_features_if_needed(edge_features)
        
        # 提取三角形索引和特征（2-单纯形）
        triangle_index = None
        triangle_features = None
        
        if simplex_2_df is not None:
            triangle_index = self._extract_triangle_connections(simplex_2_df).cpu()
            triangle_features = self._extract_triangle_features(simplex_2_df).cpu()
        
        # 创建单纯形复合体数据对象
        data = SimplexComplexData(
            x=node_features,                    # [num_nodes, node_features]
            edge_index=edge_index,              # [2, num_edges]
            edge_attr=edge_features,            # [num_edges, edge_features]
            pos=positions.cpu(),                      # [num_nodes, 3] <--- 新增坐标
            triangle_index=triangle_index,      # [3, num_triangles]
            triangle_attr=triangle_features,    # [num_triangles, triangle_features]
            material_name=material_name,
            num_nodes=len(simplex_0_df),
            num_edges=len(simplex_1_df)
        )
        
        print(f"创建单纯形复合体数据 - {material_name}:")
        print(f"  - 节点数: {data.num_nodes}, 节点特征维度: {data.x.shape[1]}")
        print(f"  - 边数: {data.num_edges}, 边特征维度: {data.edge_attr.shape[1]}")
        if triangle_features is not None and triangle_features.numel() > 0:
            print(f"  - 三角形数: {data.num_triangles}, 三角形特征维度: {data.triangle_attr.shape[1]}")
        else:
            print(f"  - 三角形数: 0 (未加载2-单纯形特征)")
        
        return data
    
    def _process_all_materials(self):
        """预处理所有材料的单纯形复合体数据"""
        print("开始预处理所有材料的单纯形复合体数据...")
        
        for material_name in self.material_names:
            try:
                data = self._create_simplex_data(material_name)
                self.processed_data.append(data)
                print(f"成功处理材料: {material_name}")
            except Exception as e:
                print(f"✗ 处理材料 {material_name} 时出错: {e}")
                continue
        
        print(f"预处理完成，成功处理 {len(self.processed_data)} 个材料")
    
    def len(self) -> int:
        """返回数据集大小"""
        return len(self.processed_data)
    
    def get(self, idx: int):
        """获取指定索引的数据"""
        if idx >= len(self.processed_data):
            raise IndexError(f"索引 {idx} 超出范围 [0, {len(self.processed_data)-1}]")
        
        data = self.processed_data[idx]
        
        if self.transform is not None:
            data = self.transform(data)
        
        # CGP数据增强：如果启用，返回(原始数据, 增强数据)的元组
        if self.enable_augmentation:
            augmented_data = self._apply_geometric_augmentation(data)
            return (data, augmented_data)
        else:
            return data
    
    def _apply_geometric_augmentation(self, data: SimplexComplexData) -> SimplexComplexData:
        """
        应用几何数据增强，生成正样本对
        Apply geometric data augmentation to generate positive pairs for CGP
        
        Args:
            data: 原始数据
            
        Returns:
            增强后的数据（作为对比学习的正样本）
        """
        # 克隆数据以避免修改原始数据
        aug_data = data.clone()
        
        # 对原子坐标添加小幅高斯噪声（模拟热振动）
        if aug_data.pos is not None:
            noise = torch.randn_like(aug_data.pos) * self.augmentation_noise_std
            aug_data.pos = aug_data.pos + noise
            
            # 重新计算依赖于坐标的边特征（前3维是向量特征）
            if hasattr(aug_data, 'edge_index') and aug_data.edge_index is not None:
                # 重新计算边向量
                edge_vectors = aug_data.pos[aug_data.edge_index[1]] - aug_data.pos[aug_data.edge_index[0]]
                
                # 更新边特征的前3维（向量部分）
                if aug_data.edge_attr is not None and aug_data.edge_attr.size(1) >= 3:
                    aug_data.edge_attr[:, :3] = edge_vectors
        
            # 重新计算依赖于坐标的三角形特征（如果存在）
            if hasattr(aug_data, 'triangle_index') and aug_data.triangle_index is not None and aug_data.triangle_attr is not None:
                # 重新计算三角形法向量
                if aug_data.triangle_index.size(1) > 0 and aug_data.triangle_attr.size(1) >= 3:
                    v1 = aug_data.pos[aug_data.triangle_index[1]] - aug_data.pos[aug_data.triangle_index[0]]
                    v2 = aug_data.pos[aug_data.triangle_index[2]] - aug_data.pos[aug_data.triangle_index[0]]
                    normals = torch.cross(v1, v2, dim=1)
                    
                    # 标准化法向量
                    normals = normals / (torch.norm(normals, dim=1, keepdim=True) + 1e-8)
                    aug_data.triangle_attr[:, :3] = normals
        
        return aug_data
    
    def get_feature_stats(self) -> Dict:
        """获取特征统计信息，用于模型设计"""
        if len(self.processed_data) == 0:
            return {}
        
        # 计算特征维度
        sample_data = self.processed_data[0]
        
        stats = {
            'num_node_features': sample_data.x.shape[1],
            'num_edge_features': sample_data.edge_attr.shape[1],
            'num_materials': len(self.processed_data),
            'avg_num_nodes': np.mean([data.num_nodes for data in self.processed_data]),
            'avg_num_edges': np.mean([data.num_edges for data in self.processed_data]),
            'material_names': [data.material_name for data in self.processed_data]
        }
        
        # 2-单纯形统计信息
        if self.load_triangles and sample_data.triangle_attr is not None and sample_data.triangle_attr.numel() > 0:
            stats['num_triangle_features'] = sample_data.triangle_attr.shape[1]
            stats['avg_num_triangles'] = np.mean([data.num_triangles for data in self.processed_data])
            stats['has_triangles'] = True
        else:
            stats['num_triangle_features'] = 0
            stats['avg_num_triangles'] = 0
            stats['has_triangles'] = False
        
        return stats

def create_simplex_dataloader(dataset: PerovskiteSimplexDataset, 
                             batch_size: int = 2, 
                             shuffle: bool = True,
                             **kwargs):
    """
    创建支持单纯形复合体的数据加载器
    
    Args:
        dataset: 钙钛矿单纯形数据集
        batch_size: 批大小
        shuffle: 是否打乱数据
        **kwargs: 其他DataLoader参数
    """
    from torch_geometric.loader import DataLoader
    
    # 当使用BatchNorm时，需要丢弃最后一个不完整的批次，以避免size=1的错误
    drop_last = kwargs.pop('drop_last', False)
    if batch_size > 1 and len(dataset) % batch_size == 1:
        print("警告: 数据集大小导致最后一个批次大小为1，将丢弃该批次 (drop_last=True)。")
        drop_last = True

    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        drop_last=drop_last,
        **kwargs
    )

def custom_collate_fn(data_list: List[SimplexComplexData]):
    """
    自定义的collate函数，用于将SimplexComplexData对象列表正确批处理。
    """
    from torch_geometric.data import Batch
    return Batch.from_data_list(data_list)

if __name__ == "__main__":
    # 测试单纯形复合体数据加载器
    print("=" * 60)
    print("测试钙钛矿单纯形复合体数据集加载器")
    print("=" * 60)
    
    # 创建数据集
    dataset = PerovskiteSimplexDataset(
        data_root="data", 
        normalize_features=True,
        load_triangles=True
    )
    
    # 打印统计信息
    stats = dataset.get_feature_stats()
    print("\n数据集统计信息:")
    for key, value in stats.items():
        if key != 'material_names':
            print(f"  {key}: {value}")
    
    # 测试数据加载
    print(f"\n测试数据加载...")
    dataloader = create_simplex_dataloader(dataset, batch_size=2, shuffle=False)
    
    if len(dataset) > 0:
        for i, batch in enumerate(dataloader):
            print(f"\n批次 {i+1}:")
            print(f"  - 批次节点数: {batch.x.shape[0]}")
            print(f"  - 批次边数: {batch.edge_index.shape[1]}")
            print(f"  - 节点特征形状: {batch.x.shape}")
            print(f"  - 边特征形状: {batch.edge_attr.shape}")
            
            if hasattr(batch, 'triangle_index') and batch.triangle_index is not None and batch.triangle_index.numel() > 0:
                print(f"  - 批次三角形数: {batch.triangle_index.shape[1]}")
                print(f"  - 三角形特征形状: {batch.triangle_attr.shape}")
            else:
                print(f"  - 三角形数: 0 (无2-单纯形数据)")
            
            print(f"  - 批次材料: {batch.material_name}")
            
            if i >= 1:  # 只测试前两个批次
                break
    else:
        print("数据为空，跳过加载器测试。")
    
    print("\n单纯形复合体数据加载器测试完成!")
