"""
单纯形神经网络(SNN)模型与哈密顿SDE动态演化
Simplicial Neural Network (SNN) Model and Hamiltonian SDE Dynamics

本文件包含TEN-FMA框架模块C的核心实现，涵盖：
1.  **阶段一：静态SNN模型 (SNN Class)**
    - 实现了基于单纯形卷积层的静态图神经网络，用于处理高阶拓扑信息。
2.  **阶段二：哈密顿SDE动态演化 (HamiltonianSDE, SNNHamiltonianDynamicsSDE Classes)**
    - 将静态SNN模型作为势能函数，构建了一个由哈密顿方程驱动的随机微分方程系统。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, TopKPooling
from .simplex_data_loader import SimplexComplexData
from .snn_layers import SimplexConvLayer
from typing import Optional, Tuple, List, Dict
import warnings


# =============================================================================
# 阶段一：静态SNN骨架 (Phase 1: Static SNN Skeleton)
# =============================================================================

# 警告：以下 SNN 模型的旧版本将被重构以严格遵循计划
# 旧的 AttentionPooling 和 MultiScalePooling 将被移除
# class AttentionPooling(...
# class MultiScalePooling(...

class SNN(nn.Module):
    """
    完整的单纯形神经网络模型 (阶段一核心)
    更正：严格实现了分层池化（Hierarchical Pooling）
    """
    def __init__(self,
                 node_input_dim: int, edge_input_dim: int, triangle_input_dim: int = 0,
                 hidden_dim: int = 128, output_dim: int = 64, num_layers: int = 4,
                 activation: str = 'swish', norm: bool = True, dropout: float = 0.1,
                 cluster_ratio: float = 0.5):
        super().__init__()
        self.node_input_dim = node_input_dim
        self.edge_input_dim = edge_input_dim
        self.triangle_input_dim = triangle_input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.use_triangles = self.triangle_input_dim > 0

        # 节点和边特征嵌入
        self.node_embedding = nn.Linear(node_input_dim, hidden_dim)
        # 假设边特征前3维是向量，其余是标量
        self.edge_embedding = nn.Linear(edge_input_dim - 3, hidden_dim - 3)
        
        if self.use_triangles:
            # 假设三角形特征前3维是法向量，其余是标量
            self.triangle_embedding = nn.Linear(triangle_input_dim - 3, hidden_dim - 3)

        # SNN 卷积层
        self.layers = nn.ModuleList([
            SimplexConvLayer(
                node_dim=hidden_dim, 
                edge_dim=hidden_dim, 
                triangle_dim=hidden_dim if self.use_triangles else 0,
                hidden_dim=hidden_dim,
                activation=activation, 
                norm=norm,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # 输出层，将节点特征投影到最终的 output_dim
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        # 分层池化第一步: 局部团簇池化 (使用 TopKPooling)
        self.cluster_pooling = TopKPooling(output_dim, ratio=cluster_ratio)
        
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def embed_edges(self, edge_attr: torch.Tensor) -> torch.Tensor:
        """Helper function to embed edge scalar features."""
        edge_vectors, edge_scalars = edge_attr[:, :3], edge_attr[:, 3:]
        return torch.cat([edge_vectors, self.edge_embedding(edge_scalars)], dim=-1)

    def embed_triangles(self, triangle_attr: torch.Tensor) -> torch.Tensor:
        """Helper function to embed triangle scalar features."""
        if not self.use_triangles or triangle_attr is None or triangle_attr.numel() == 0:
            return None
        # 假设三角形特征也有向量和标量部分
        triangle_vectors, triangle_scalars = triangle_attr[:, :3], triangle_attr[:, 3:]
        return torch.cat([triangle_vectors, self.triangle_embedding(triangle_scalars)], dim=-1)

    def forward(self, data: SimplexComplexData) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播，实现分层池化.
        返回:
            final_node_features (torch.Tensor): (num_nodes, output_dim), 池化前的最终节点特征
            global_embedding (torch.Tensor): (batch_size, output_dim), 分层池化后的全局图嵌入
        """
        node_features = self.node_embedding(data.x)
        edge_features = self.embed_edges(data.edge_attr)
        triangle_features = self.embed_triangles(getattr(data, 'triangle_attr', None))
        
        for layer in self.layers:
            # 修正：传递所有必需的参数，包括原子坐标和高阶单纯形信息
            node_features, edge_features, triangle_features = layer(
                node_features=node_features,
                edge_features=edge_features,
                edge_index=data.edge_index,
                positions=data.pos,
                triangle_features=triangle_features,
                triangle_index=getattr(data, 'triangle_index', None)
            )
        
        # 应用最终的线性层以匹配 output_dim
        final_node_features = self.output_layer(node_features)

        # --- 分层池化 (Hierarchical Pooling) ---
        # 步骤 1: 局部团簇池化 (使用TopKPooling筛选代表性节点)
        cluster_features, _, _, cluster_batch, _, _ = self.cluster_pooling(
            final_node_features, data.edge_index, batch=data.batch
        )
        
        # 步骤 2: 全局池化 (对团簇代表节点进行全局平均)
        global_embedding = global_mean_pool(cluster_features, cluster_batch)

        # 为GraphMAE2返回池化前的完整节点特征，为最终输出返回全局嵌入
        return final_node_features, global_embedding


class HamiltonianSDE(nn.Module):
    """
    哈密顿随机微分方程 (SDE) 的漂移项和扩散项模块。
    将SNN模型封装为哈密顿量 H(q, p)，并计算其梯度作为漂移函数。
    """
    noise_type = "diagonal"
    sde_type = "stratonovich"

    def __init__(self, snn_model: SNN):
        super().__init__()
        self.snn_model = snn_model
        self.potential_head = nn.Sequential(
            nn.Linear(snn_model.output_dim, snn_model.output_dim // 2),
            nn.SiLU(), nn.Linear(snn_model.output_dim // 2, 1)
        )
        self.static_data = None

    def _compute_hamiltonian(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        q = q.clone().contiguous()
        p = p.clone().contiguous()

        current_data = SimplexComplexData(
            x=self.static_data.x, pos=q, edge_index=self.static_data.edge_index,
            edge_attr=self.static_data.edge_attr,
            triangle_index=getattr(self.static_data, 'triangle_index', None),
            triangle_attr=getattr(self.static_data, 'triangle_attr', None),
            batch=getattr(self.static_data, 'batch', None)
        ).to(q.device)
        
        global_embedding, _ = self.snn_model(current_data)
        potential_energy = self.potential_head(global_embedding).sum()
        kinetic_energy = 0.5 * torch.sum(p * p)
        return potential_energy + kinetic_energy

    def f(self, t: float, y: torch.Tensor) -> torch.Tensor:
        """SDE漂移函数 f(t, y) = (dH/dp, -dH/dq)"""
        with torch.enable_grad():
            y = y.requires_grad_(True)
            q, p = torch.chunk(y, 2, dim=-1)
            
            hamiltonian = self._compute_hamiltonian(q.view_as(self.static_data.pos), p.view_as(self.static_data.pos))
            
            grad_h = torch.autograd.grad(hamiltonian, y, grad_outputs=torch.ones_like(hamiltonian))[0]
            grad_q, grad_p = torch.chunk(grad_h, 2, dim=-1)
            result = torch.cat([grad_p, -grad_q], dim=-1)
            
            return result.contiguous()

    def g(self, t: float, y: torch.Tensor) -> torch.Tensor:
        """SDE扩散函数 g(t, y)"""
        # y shape: (batch, state_dim)
        batch_size = y.shape[0] if y.dim() == 2 else 1
        q_dim = y.shape[-1] // 2
        diffusion_q = torch.zeros(q_dim, device=y.device)
        diffusion_p = torch.full((q_dim,), 1e-3, device=y.device)
        diffusion_vec = torch.cat([diffusion_q, diffusion_p]).contiguous()  # (state_dim,)
        return diffusion_vec.unsqueeze(0).repeat(batch_size, 1).contiguous()

class SNNHamiltonianDynamicsSDE(nn.Module):
    """
    基于哈密顿SDE的SNN动力学演化模型 (阶段二核心)

    TEN-FMA框架的核心动力引擎，采用朗之万蛙跳法辛积分器实现：
    - 物理严谨：模拟系统与热浴的相互作用
    - 梯度可控：端到端可微分，支持模型训练
    - 纯结构驱动：符合框架设计理念
    """
    def __init__(self,
                 snn_model: SNN, integration_time: float = 1.0,
                 integration_steps: int = 10,
                 temperature: float = 0.1, friction: float = 0.01):
        super().__init__()
        self.sde_module = HamiltonianSDE(snn_model)
        self.integration_time = integration_time
        self.integration_steps = integration_steps
        self.temperature = temperature
        self.friction = friction
        # 玻尔兹曼常数 k_B 设为1
        self.register_buffer(
            "ts", torch.linspace(0, self.integration_time, self.integration_steps)
        )
        
    def forward(self, data: SimplexComplexData) -> Tuple[torch.Tensor, torch.Tensor, List[float], List[float], List[float], torch.Tensor]:
        """
        前向传播：使用BAOAB朗之万蛙跳法辛积分器求解哈密顿SDE
        Forward pass: Solve Hamiltonian SDE using the BAOAB Langevin Leapfrog symplectic integrator.
        
        Returns:
            q_final (torch.Tensor): Final positions.
            p_final (torch.Tensor): Final momenta.
            hamiltonian_history (List[float]): Hamiltonian value at each step.
            kinetic_history (List[float]): Kinetic energy at each step.
            potential_history (List[float]): Potential energy at each step.
            trajectory_history (torch.Tensor): Atom positions at each step [steps, num_atoms, 3].
        """
        q = data.pos
        p = torch.randn_like(q) * torch.sqrt(torch.tensor(self.temperature, device=q.device))

        dt = self.integration_time / self.integration_steps
        
        device = q.device
        c1 = torch.exp(torch.tensor(-self.friction * dt, device=device))
        c2 = torch.sqrt((1 - c1 * c1) * self.temperature)

        hamiltonian_history = []
        kinetic_history = []
        potential_history = []
        trajectory_history = [q.clone().cpu()]

        # BAOAB 朗之万积分器 (g-BAOAB "middle" scheme)
        self.sde_module.static_data = data
        
        for i in range(self.integration_steps):
            # --- B Step (half) ---
            with torch.enable_grad():
                q_clone = q.clone().requires_grad_(True)
                hamiltonian = self.sde_module._compute_hamiltonian(q_clone, p)
                
                kinetic_energy = 0.5 * torch.sum(p * p)
                potential_energy = hamiltonian - kinetic_energy
                
                hamiltonian_history.append(hamiltonian.item())
                kinetic_history.append(kinetic_energy.item())
                potential_history.append(potential_energy.item())
                
                grad_q = torch.autograd.grad(hamiltonian, q_clone, grad_outputs=torch.ones_like(hamiltonian), create_graph=True)[0]
            
            p_half = p - 0.5 * dt * grad_q
            
            # --- A Step (half) ---
            # 假设质量为1，则 p = v
            q_half = q + 0.5 * dt * p_half
            
            # --- O Step (full) ---
            p_half_thermostatted = c1 * p_half + c2 * torch.randn_like(p_half)
            
            # --- A Step (half) ---
            q = q_half + 0.5 * dt * p_half_thermostatted
            trajectory_history.append(q.clone().cpu())
            
            # --- B Step (half) ---
            self.sde_module.static_data.pos = q # 更新位置
            with torch.enable_grad():
                q_clone = q.clone().requires_grad_(True)
                hamiltonian_for_p = self.sde_module._compute_hamiltonian(q_clone, p_half_thermostatted)
                grad_q = torch.autograd.grad(hamiltonian_for_p, q_clone, grad_outputs=torch.ones_like(hamiltonian_for_p), create_graph=True)[0]
            
            p = p_half_thermostatted - 0.5 * dt * grad_q
            
        trajectory_history_tensor = torch.stack(trajectory_history, dim=0)

        return q, p, hamiltonian_history, kinetic_history, potential_history, trajectory_history_tensor


    def get_final_embeddings(self, data: SimplexComplexData) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        执行完整的SDE演化，并返回最终的节点特征和全局嵌入。
        此方法用于外部调用（如SSL任务）。
        """
        # 1. SDE演化得到最终的原子位置
        q_final, _, _, _, _, _ = self.forward(data)
        
        # 2. 用最终的原子位置更新图数据
        final_data = data.clone()
        final_data.pos = q_final

        # 3. 将更新后的图再次通过SNN模型，以获取最终的嵌入
        # SNN模型现在返回节点特征和全局嵌入
        node_features, global_embedding = self.sde_module.snn_model(final_data)
        return node_features, global_embedding

def create_snn_model(node_input_dim: int, edge_input_dim: int,
                     triangle_input_dim: int = 0, **kwargs) -> SNN:
    """便捷函数：创建SNN模型"""
    default_params = {
        'hidden_dim': 128, 'output_dim': 64, 'num_layers': 4,
        'activation': 'swish', 'norm': True, 'dropout': 0.1,
        'cluster_ratio': 0.5 # 新增参数
    }
    default_params.update(kwargs)
    return SNN(
        node_input_dim=node_input_dim, edge_input_dim=edge_input_dim,
        triangle_input_dim=triangle_input_dim, **default_params
    )