"""
单纯形神经网络(SNN)卷积层实现
Simplicial Neural Network (SNN) Convolutional Layers Implementation

遵循GUIDELINES-ModelC.md阶段二要求：实现支持0-1-2单纯形的高阶消息传递
Following Stage 2 requirements: Implement higher-order message passing for 0-1-2 simplices

基于论文:
- "Simplicial Neural Networks" (Bunch et al., 2020)
- "Weisfeiler and Leman Go Topological" (Morris et al., 2020)
- E(3) Equivariant Graph Neural Networks (Satorras et al., 2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from typing import Optional, Tuple, Dict, List
import math
import numpy as np

class SimplexMessagePassing(nn.Module):
    """
    单纯形消息传递基础类
    Base class for simplicial message passing
    
    实现不同维度单纯形之间的消息传递机制
    Implements message passing between simplices of different dimensions
    """
    
    def __init__(self, 
                 node_dim: int,
                 edge_dim: int, 
                 triangle_dim: int,
                 hidden_dim: int = 128,
                 activation: str = 'swish',
                 norm: bool = True):
        """
        初始化单纯形消息传递层
        
        Args:
            node_dim: 节点特征维度 (0-simplex)
            edge_dim: 边特征维度 (1-simplex)  
            triangle_dim: 三角形特征维度 (2-simplex)
            hidden_dim: 隐藏层维度
            activation: 激活函数类型
            norm: 是否使用层归一化
        """
        super().__init__()
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.triangle_dim = triangle_dim
        self.hidden_dim = hidden_dim
        
        # 激活函数
        if activation == 'swish':
            self.activation = nn.SiLU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise ValueError(f"不支持的激活函数: {activation}")
        
        # 归一化层
        if norm:
            self.node_norm = nn.LayerNorm(node_dim)
            if edge_dim > 3:
                self.edge_norm = nn.LayerNorm(edge_dim - 3) # 只对标量部分进行归一化
            if triangle_dim > 3:
                self.triangle_norm = nn.LayerNorm(triangle_dim - 3) # 只对标量部分进行归一化
        else:
            self.node_norm = nn.Identity()
            self.edge_norm = nn.Identity()
            if triangle_dim > 0:
                self.triangle_norm = nn.Identity()
    
    def reset_parameters(self):
        """重置网络参数"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

class NodeToEdgeMessagePassing(SimplexMessagePassing):
    """
    节点到边的消息传递 (0-simplex to 1-simplex)
    Node to Edge Message Passing
    
    将节点特征聚合到边特征中
    Aggregates node features to edge features
    """
    
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int = 128, **kwargs):
        super().__init__(node_dim, edge_dim, 0, hidden_dim, **kwargs)
        
        # 消息计算网络：[node_i_features || node_j_features || edge_scalar_features || relative_dist_sq] -> hidden
        self.message_net = nn.Sequential(
            nn.Linear(2 * node_dim + (edge_dim - 3) + 1, hidden_dim), # +1 for distance, -3 for vector
            self.activation,
            nn.Linear(hidden_dim, hidden_dim),
            self.activation
        )
        
        # 边更新网络：只更新边的标量部分
        self.update_net = nn.Sequential(
            nn.Linear((edge_dim - 3) + hidden_dim, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, edge_dim - 3)
        )
    
    def forward(self, 
                node_features: torch.Tensor,     # [num_nodes, node_dim]
                edge_features: torch.Tensor,     # [num_edges, edge_dim]
                edge_index: torch.Tensor,        # [2, num_edges]
                positions: torch.Tensor          # [num_nodes, 3]
                ) -> torch.Tensor:
        """
        前向传播：节点到边的消息传递
        
        Args:
            node_features: 节点特征
            edge_features: 边特征
            edge_index: 边索引
            positions: 节点坐标
            
        Returns:
            updated_edge_features: 更新后的边特征
        """
        row, col = edge_index[0], edge_index[1]  # source, target nodes
        
        # 计算相对距离的平方
        relative_pos = positions[row] - positions[col]
        relative_dist_sq = (relative_pos ** 2).sum(dim=-1, keepdim=True)
        
        # 构建消息：只使用标量特征
        messages = torch.cat([
            node_features[row],      # 源节点特征 (标量)
            node_features[col],      # 目标节点特征 (标量)
            edge_features[:, 3:],    # 边的标量部分
            relative_dist_sq         # 相对距离平方 (标量)
        ], dim=-1)
        
        # 计算消息
        processed_messages = self.message_net(messages)
        
        # 分离边特征的向量和标量部分
        edge_vectors = edge_features[:, :3]
        edge_scalars = edge_features[:, 3:]
        
        # 更新边的标量部分
        update_input = torch.cat([edge_scalars, processed_messages], dim=-1)
        updated_edge_scalars = self.update_net(update_input)
        
        # 残差连接和归一化（只对标量部分）
        updated_edge_scalars_res = edge_scalars + updated_edge_scalars
        if hasattr(self, 'edge_norm'):
            updated_edge_scalars_res = self.edge_norm(updated_edge_scalars_res)

        
        # 重新组合特征
        updated_edge_features = torch.cat([edge_vectors, updated_edge_scalars_res], dim=-1)
        
        return updated_edge_features

class EdgeToTriangleMessagePassing(SimplexMessagePassing):
    """
    边到三角形的消息传递 (1-simplex to 2-simplex)
    Edge to Triangle Message Passing
    
    将边特征聚合到三角形特征中
    Aggregates edge features to triangle features
    """
    
    def __init__(self, edge_dim: int, triangle_dim: int, hidden_dim: int = 128, **kwargs):
        super().__init__(0, edge_dim, triangle_dim, hidden_dim, **kwargs)
        
        # 消息计算网络：[edge_ij_scalars || edge_ik_scalars || edge_jk_scalars || triangle_scalars] -> hidden
        self.message_net = nn.Sequential(
            nn.Linear(3 * (edge_dim - 3) + (triangle_dim - 3), hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, hidden_dim),
            self.activation
        )
        
        # 三角形更新网络: 只更新三角形的标量部分
        self.update_net = nn.Sequential(
            nn.Linear((triangle_dim - 3) + hidden_dim, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, triangle_dim - 3)
        )
    
    def forward(self,
                edge_features: torch.Tensor,      # [num_edges, edge_dim]
                triangle_features: torch.Tensor,  # [num_triangles, triangle_dim]
                edge_index: torch.Tensor,         # [2, num_edges]
                triangle_index: torch.Tensor      # [3, num_triangles]
                ) -> torch.Tensor:
        """
        前向传播：边到三角形的消息传递
        
        Args:
            edge_features: 边特征
            triangle_features: 三角形特征
            edge_index: 边索引
            triangle_index: 三角形索引
            
        Returns:
            updated_triangle_features: 更新后的三角形特征
        """
        if triangle_features is None or triangle_features.size(0) == 0:
            return triangle_features
        
        # 为每个三角形找到对应的三条边
        triangle_edge_features = self._get_triangle_edge_features(
            edge_features, edge_index, triangle_index
        )
        
        # 如果无法获取边特征，返回原始三角形特征
        if triangle_edge_features is None:
            return triangle_features
        
        # (新增) 从特征中剥离向量部分，只使用标量
        num_triangles = triangle_features.size(0)
        # 边特征的标量部分
        triangle_edge_scalars = triangle_edge_features.view(
            num_triangles, 3, self.edge_dim)[:, :, 3:].reshape(num_triangles, -1)
        # 三角形特征的标量部分
        triangle_feature_scalars = triangle_features[:, 3:]
        
        # 构建消息：连接三条边的标量特征和三角形的标量特征
        messages = torch.cat([
            triangle_edge_scalars,
            triangle_feature_scalars
        ], dim=-1)
        
        # 计算消息
        processed_messages = self.message_net(messages)
        
        # 分离三角形特征的向量和标量部分
        triangle_vectors = triangle_features[:, :3]
        triangle_scalars = triangle_features[:, 3:]
        
        # 更新三角形的标量部分
        update_input = torch.cat([triangle_scalars, processed_messages], dim=-1)
        updated_triangle_scalars = self.update_net(update_input)
        
        # 残差连接和归一化（只对标量部分）
        updated_triangle_scalars_res = triangle_scalars + updated_triangle_scalars
        if hasattr(self, 'triangle_norm'):
            updated_triangle_scalars_res = self.triangle_norm(updated_triangle_scalars_res)

        # 重新组合特征
        updated_triangle_features = torch.cat([triangle_vectors, updated_triangle_scalars_res], dim=-1)
        
        return updated_triangle_features
    
    def _get_triangle_edge_features(self,
                                   edge_features: torch.Tensor,
                                   edge_index: torch.Tensor,
                                   triangle_index: torch.Tensor) -> Optional[torch.Tensor]:
        """
        获取每个三角形对应的三条边的特征
        
        Args:
            edge_features: 边特征 [num_edges, edge_dim]
            edge_index: 边索引 [2, num_edges]
            triangle_index: 三角形索引 [3, num_triangles]
            
        Returns:
            triangle_edge_features: 每个三角形的边特征 [num_triangles, 3*edge_dim]
        """
        if triangle_index.size(1) == 0:
            return None
        
        device = edge_features.device
        num_triangles = triangle_index.size(1)
        edge_dim = edge_features.size(1)
        
        # 创建边索引的查找表 (无向图，需要考虑双向)
        edge_index_cpu = edge_index.cpu()
        edge_dict = {}
        for i, (src, tgt) in enumerate(edge_index_cpu.t()):
            src_val, tgt_val = int(src), int(tgt)
            edge_dict[(min(src_val, tgt_val), max(src_val, tgt_val))] = i
        
        triangle_edge_features_list = []
        
        for tri_idx in range(num_triangles):
            v0, v1, v2 = triangle_index[:, tri_idx].cpu().tolist()
            
            # 三角形的三条边 (确保一致的排序)
            edges_in_triangle = [
                (min(v0, v1), max(v0, v1)),  # edge 01
                (min(v1, v2), max(v1, v2)),  # edge 12
                (min(v2, v0), max(v2, v0))   # edge 20
            ]
            
            triangle_edges = []
            for edge_tuple in edges_in_triangle:
                if edge_tuple in edge_dict:
                    edge_idx = edge_dict[edge_tuple]
                    triangle_edges.append(edge_features[edge_idx])
                else:
                    # 如果找不到边，用零向量填充
                    triangle_edges.append(torch.zeros(edge_dim, device=device))
            
            # 连接三条边的特征
            triangle_edge_feature = torch.cat(triangle_edges, dim=0)  # [3*edge_dim]
            triangle_edge_features_list.append(triangle_edge_feature)
        
        if not triangle_edge_features_list:
            return None
            
        return torch.stack(triangle_edge_features_list, dim=0)  # [num_triangles, 3*edge_dim]

class TriangleToEdgeMessagePassing(SimplexMessagePassing):
    """
    三角形到边的消息传递 (2-simplex to 1-simplex)
    Triangle to Edge Message Passing
    
    将三角形特征反馈到边特征中
    Feeds triangle features back to edge features
    """
    
    def __init__(self, edge_dim: int, triangle_dim: int, hidden_dim: int = 128, **kwargs):
        super().__init__(0, edge_dim, triangle_dim, hidden_dim, **kwargs)
        
        # 消息计算网络: 只使用三角形的标量部分
        self.message_net = nn.Sequential(
            nn.Linear(triangle_dim - 3, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, hidden_dim),
            self.activation
        )
        
        # 边更新网络: 只更新边的标量部分
        self.update_net = nn.Sequential(
            nn.Linear((edge_dim - 3) + hidden_dim, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, edge_dim - 3)
        )
    
    def forward(self,
                edge_features: torch.Tensor,      # [num_edges, edge_dim]
                triangle_features: torch.Tensor,  # [num_triangles, triangle_dim]
                edge_index: torch.Tensor,         # [2, num_edges]
                triangle_index: torch.Tensor      # [3, num_triangles]
                ) -> torch.Tensor:
        """
        前向传播：三角形到边的消息传递
        """
        if triangle_features is None or triangle_features.size(0) == 0:
            return edge_features
        
        # 为每条边收集来自相关三角形的消息
        edge_triangle_messages = self._aggregate_triangle_messages_to_edges(
            edge_features, triangle_features, edge_index, triangle_index
        )
        
        # 分离边特征的向量和标量部分
        edge_vectors = edge_features[:, :3]
        edge_scalars = edge_features[:, 3:]
        
        # 更新边的标量部分
        update_input = torch.cat([edge_scalars, edge_triangle_messages], dim=-1)
        updated_edge_scalars = self.update_net(update_input)
        
        # 残差连接和归一化（只对标量部分）
        updated_edge_scalars_res = edge_scalars + updated_edge_scalars
        if hasattr(self, 'edge_norm'):
            updated_edge_scalars_res = self.edge_norm(updated_edge_scalars_res)

        # 重新组合特征
        updated_edge_features = torch.cat([edge_vectors, updated_edge_scalars_res], dim=-1)
        
        return updated_edge_features
    
    def _aggregate_triangle_messages_to_edges(self,
                                             edge_features: torch.Tensor,
                                             triangle_features: torch.Tensor,
                                             edge_index: torch.Tensor,
                                             triangle_index: torch.Tensor) -> torch.Tensor:
        """
        将三角形消息聚合到边上
        """
        device = edge_features.device
        num_edges = edge_features.size(0)
        hidden_dim = self.hidden_dim
        
        # 创建边到三角形的映射
        edge_to_triangles = self._build_edge_to_triangle_mapping(edge_index, triangle_index)
        
        # 预计算所有三角形特征的消息 (只使用标量部分)
        triangle_messages = self.message_net(triangle_features[:, 3:])  # [num_triangles, hidden_dim]
        
        # 为每条边聚合来自相关三角形的消息
        aggregated_messages = torch.zeros(num_edges, hidden_dim, device=device)
        
        for edge_idx, triangle_indices in edge_to_triangles.items():
            if len(triangle_indices) > 0:
                # 平均聚合相关三角形的消息
                relevant_messages = triangle_messages[triangle_indices]  # [num_relevant_triangles, hidden_dim]
                aggregated_message = torch.mean(relevant_messages, dim=0)  # [hidden_dim]
                aggregated_messages[edge_idx] = aggregated_message
        
        return aggregated_messages
    
    def _build_edge_to_triangle_mapping(self,
                                       edge_index: torch.Tensor,
                                       triangle_index: torch.Tensor) -> Dict[int, List[int]]:
        """
        构建边到三角形的映射关系
        """
        edge_index_cpu = edge_index.cpu()
        edge_to_triangles = {i: [] for i in range(edge_index.size(1))}
        
        # 创建边索引的查找表
        edge_dict = {}
        for i, (src, tgt) in enumerate(edge_index_cpu.t()):
            src_val, tgt_val = int(src), int(tgt)
            edge_dict[(min(src_val, tgt_val), max(src_val, tgt_val))] = i
        
        # 遍历每个三角形，找到其对应的边
        triangle_index_cpu = triangle_index.cpu()
        for tri_idx in range(triangle_index.size(1)):
            v0, v1, v2 = triangle_index_cpu[:, tri_idx].tolist()
            
            # 三角形的三条边
            edges_in_triangle = [
                (min(v0, v1), max(v0, v1)),
                (min(v1, v2), max(v1, v2)),
                (min(v2, v0), max(v2, v0))
            ]
            
            for edge_tuple in edges_in_triangle:
                if edge_tuple in edge_dict:
                    edge_idx = edge_dict[edge_tuple]
                    edge_to_triangles[edge_idx].append(tri_idx)
        
        return edge_to_triangles

class EdgeToNodeMessagePassing(SimplexMessagePassing):
    """
    边到节点的消息传递 (1-simplex to 0-simplex)
    Edge to Node Message Passing
    
    将边特征聚合到节点特征中
    Aggregates edge features to node features
    """
    
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int = 128, **kwargs):
        super().__init__(node_dim, edge_dim, 0, hidden_dim, **kwargs)
        
        # 消息计算网络: [node_i || edge_ij_scalars || relative_dist_sq] -> hidden
        self.message_net = nn.Sequential(
            nn.Linear(node_dim + (edge_dim - 3) + 1, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, hidden_dim),
            self.activation
        )
        
        # 节点更新网络
        self.update_net = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, node_dim)
        )
    
    def forward(self,
                node_features: torch.Tensor,     # [num_nodes, node_dim]
                edge_features: torch.Tensor,     # [num_edges, edge_dim]
                edge_index: torch.Tensor,        # [2, num_edges]
                positions: torch.Tensor          # [num_nodes, 3]
                ) -> torch.Tensor:
        """
        前向传播：边到节点的消息传递
        """
        row, col = edge_index[0], edge_index[1]
        
        # 计算相对距离
        relative_pos = positions[row] - positions[col]
        relative_dist_sq = (relative_pos ** 2).sum(dim=-1, keepdim=True)
        
        # 构建消息 (从 col -> row), 只使用标量特征
        message_input_to_row = torch.cat([node_features[col], edge_features[:, 3:], relative_dist_sq], dim=-1)
        messages_to_row = self.message_net(message_input_to_row)
        
        # 构建消息 (从 row -> col), 只使用标量特征
        message_input_to_col = torch.cat([node_features[row], edge_features[:, 3:], relative_dist_sq], dim=-1)
        messages_to_col = self.message_net(message_input_to_col)
        
        # 聚合消息到节点
        num_nodes = node_features.size(0)
        aggregated_messages = torch.zeros(num_nodes, self.hidden_dim, 
                                        device=node_features.device)
        
        aggregated_messages.index_add_(0, row, messages_to_row.to(aggregated_messages.dtype))
        aggregated_messages.index_add_(0, col, messages_to_col.to(aggregated_messages.dtype))
        
        # 更新节点特征
        node_input = torch.cat([node_features, aggregated_messages], dim=-1)
        updated_node_features = self.update_net(node_input)
        
        # 残差连接和归一化
        updated_node_features = self.node_norm(node_features + updated_node_features)
        
        return updated_node_features

class SimplexConvLayer(nn.Module):
    """
    完整的单纯形卷积层
    Complete Simplicial Convolutional Layer
    
    集成所有维度间的消息传递：0-simplex ↔ 1-simplex ↔ 2-simplex
    Integrates all inter-dimensional message passing
    """
    
    def __init__(self,
                 node_dim: int,
                 edge_dim: int,
                 triangle_dim: int = 0,
                 hidden_dim: int = 128,
                 activation: str = 'swish',
                 norm: bool = True,
                 dropout: float = 0.1):
        """
        初始化单纯形卷积层
        
        Args:
            node_dim: 节点特征维度
            edge_dim: 边特征维度
            triangle_dim: 三角形特征维度 (0表示不使用三角形)
            hidden_dim: 隐藏层维度
            activation: 激活函数
            norm: 是否使用归一化
            dropout: Dropout概率
        """
        super().__init__()
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.triangle_dim = triangle_dim
        self.use_triangles = triangle_dim > 0
        
        # 消息传递组件
        self.node_to_edge = NodeToEdgeMessagePassing(
            node_dim, edge_dim, hidden_dim, activation=activation, norm=norm
        )
        
        self.edge_to_node = EdgeToNodeMessagePassing(
            node_dim, edge_dim, hidden_dim, activation=activation, norm=norm
        )
        
        if self.use_triangles:
            self.edge_to_triangle = EdgeToTriangleMessagePassing(
                edge_dim, triangle_dim, hidden_dim, activation=activation, norm=norm
            )
            
            self.triangle_to_edge = TriangleToEdgeMessagePassing(
                edge_dim, triangle_dim, hidden_dim, activation=activation, norm=norm
            )
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
    
    def forward(self,
                node_features: torch.Tensor,
                edge_features: torch.Tensor,
                edge_index: torch.Tensor,
                positions: torch.Tensor, # <--- 新增
                triangle_features: Optional[torch.Tensor] = None,
                triangle_index: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播：执行完整的单纯形消息传递
        
        Args:
            node_features: 节点特征 [num_nodes, node_dim]
            edge_features: 边特征 [num_edges, edge_dim]
            edge_index: 边索引 [2, num_edges]
            positions: 节点坐标 [num_nodes, 3]
            triangle_features: 三角形特征 [num_triangles, triangle_dim] (可选)
            triangle_index: 三角形索引 [3, num_triangles] (可选)
            
        Returns:
            (updated_nodes, updated_edges, updated_triangles)
        """
        # 保存输入用于残差连接
        node_input = node_features
        edge_input = edge_features
        triangle_input = triangle_features
        
        # 第一阶段：向上消息传递 (0→1→2)
        # 节点 → 边
        h_edge = self.node_to_edge(node_features, edge_features, edge_index, positions)
        h_edge = self.dropout(h_edge)
        
        # 边 → 三角形 (如果存在)
        h_triangle = triangle_features
        if self.use_triangles and h_triangle is not None and triangle_index is not None:
            h_triangle = self.edge_to_triangle(
                h_edge, h_triangle, edge_index, triangle_index
            )
            h_triangle = self.dropout(h_triangle)
        
        # 第二阶段：向下消息传递 (2→1→0)
        # 三角形 → 边 (如果存在)
        if self.use_triangles and h_triangle is not None and triangle_index is not None:
            h_edge = self.triangle_to_edge(
                h_edge, h_triangle, edge_index, triangle_index
            )
            h_edge = self.dropout(h_edge)
        
        # 边 → 节点
        h_node = self.edge_to_node(node_features, h_edge, edge_index, positions)
        h_node = self.dropout(h_node)
        
        return h_node, h_edge, h_triangle
    
    def reset_parameters(self):
        """重置所有参数"""
        self.node_to_edge.reset_parameters()
        self.edge_to_node.reset_parameters()
        if self.use_triangles:
            self.edge_to_triangle.reset_parameters()
            self.triangle_to_edge.reset_parameters()
