"""
Self-Supervised Learning (SSL) Modules for TEN-FMA Framework
自监督学习（SSL）模块

本文件包含为TEN-FMA框架设计的自监督学习任务的核心实现。
当前实现了 GraphMAE2 任务。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from typing import Tuple, Dict

from .snn_model import SNNHamiltonianDynamicsSDE
from .snn_layers import SimplexConvLayer
from .simplex_data_loader import SimplexComplexData

# =============================================================================
# GraphMAE2: 解码增强的掩码图自编码器
# =============================================================================

class EMA:
    """
    指数移动平均 (Exponential Moving Average)
    用于平滑更新教师模型的权重，使其比学生模型更稳定。
    """
    def __init__(self, beta: float):
        super().__init__()
        self.beta = beta

    def update_average(self, old_avg, new_val):
        if old_avg is None:
            return new_val
        return old_avg * self.beta + (1 - self.beta) * new_val

def update_moving_average(ema_updater: EMA, ma_model: nn.Module, current_model: nn.Module):
    """
    执行教师模型（ma_model）的EMA更新。
    """
    for ma_params, current_params in zip(ma_model.parameters(), current_model.parameters()):
        old_ma_val, current_val = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_ma_val, current_val)

class GraphMAE2_SSL(nn.Module):
    """
    GraphMAE2 自监督学习模块
    将SDE模型封装为学生-教师架构，并实现GraphMAE2的两个核心自监督任务。
    """
    def __init__(self,
                 student_encoder,
                 mask_rate: float = 0.75,
                 remask_rate: float = 0.5,
                 num_remask_views: int = 4,
                 latent_loss_lambda: float = 1.0,
                 ema_decay: float = 0.999,
                 gamma: float = 2.0):
        super().__init__()
        
        self.mask_rate = mask_rate
        self.remask_rate = remask_rate
        self.num_remask_views = num_remask_views
        self.latent_loss_lambda = latent_loss_lambda
        self.gamma = gamma

        # 1. 初始化学生和教师模型
        self.student_encoder = student_encoder
        self.teacher_encoder = self._create_teacher_encoder(student_encoder)
        self.ema_updater = EMA(ema_decay)

        # 2. 获取特征维度信息 (已移除E3SNN逻辑)
        snn_model = student_encoder.sde_module.snn_model
        node_dim = snn_model.node_input_dim
        edge_dim = snn_model.edge_input_dim
        hidden_dim = snn_model.hidden_dim

        # 3. 初始化掩码Token (可学习参数)
        self.node_mask_token = nn.Parameter(torch.randn(1, node_dim))
        self.decoder_mask_token = nn.Parameter(torch.randn(1, hidden_dim))

        # 4. 初始化解码器
        # 新增：为解码器创建一个边嵌入层，与主编码器SNN中的逻辑保持一致
        self.decoder_edge_embedding = nn.Linear(edge_dim - 3, hidden_dim - 3)

        # 任务1：输入特征重构解码器 (轻量级SNN层)
        self.input_recon_decoder = SimplexConvLayer(
            node_dim=hidden_dim, edge_dim=hidden_dim, triangle_dim=0, # 简化，不处理高阶
            hidden_dim=hidden_dim, activation='relu', norm=False
        )
        self.decoder_to_input_head = nn.Linear(hidden_dim, node_dim)

        # 任务2：潜在表征预测解码器 (MLP)
        self.latent_pred_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # 冻结教师网络梯度
        for param in self.teacher_encoder.parameters():
            param.requires_grad = False

    def _create_teacher_encoder(self, encoder: nn.Module) -> nn.Module:
        """创建一个与学生编码器结构相同但不共享权重的教师编码器"""
        t_encoder = deepcopy(encoder)
        return t_encoder

    def update_teacher(self):
        """执行教师模型的EMA更新"""
        update_moving_average(self.ema_updater, self.teacher_encoder, self.student_encoder)

    def forward(self, data: SimplexComplexData) -> Tuple[torch.Tensor, Dict]:
        """
        执行完整的GraphMAE2前向传播和损失计算
        """
        device = data.x.device
        
        # --- 1. 准备数据和掩码 ---
        # ... masking logic ...
        mask_nodes = torch.bernoulli(torch.full((data.num_nodes,), self.mask_rate, device=device)).bool()
        masked_data = data.clone()
        masked_data.x[mask_nodes] = self.node_mask_token.expand_as(masked_data.x[mask_nodes])
        
        # --- 2. 学生编码器处理掩码数据 ---
        # 修正：调用新的 get_final_embeddings 接口，并只取我们需要的节点特征
        student_encoded_nodes, _ = self.student_encoder.get_final_embeddings(masked_data)
        
        # --- 任务1: 多视角输入特征重构 ---
        # 修正：为解码器准备嵌入后的边特征
        edge_vectors, edge_scalars = masked_data.edge_attr[:, :3], masked_data.edge_attr[:, 3:]
        decoder_edge_features = torch.cat([edge_vectors, self.decoder_edge_embedding(edge_scalars)], dim=-1)

        recon_losses = []
        for _ in range(self.num_remask_views):
            # 随机重掩码
            num_remask_nodes = int(self.remask_rate * data.num_nodes)
            remask_perm = torch.randperm(data.num_nodes, device=device)
            remask_nodes = remask_perm[:num_remask_nodes]
            
            remasked_student_nodes = student_encoded_nodes.clone()
            remasked_student_nodes[remask_nodes] = self.decoder_mask_token.to(remasked_student_nodes.dtype).expand_as(remasked_student_nodes[remask_nodes])
            
            # 解码
            decoded_nodes, _, _ = self.input_recon_decoder(
                node_features=remasked_student_nodes,
                edge_features=decoder_edge_features, # 修正：使用嵌入后的边特征
                edge_index=masked_data.edge_index,
                positions=masked_data.pos,
            )
            recon_features = self.decoder_to_input_head(decoded_nodes[mask_nodes])
            
            # 计算损失 (Scaled Cosine Error)
            # 确保输入维度正确
            assert recon_features.shape == data.x[mask_nodes].shape, f"Shape mismatch: {recon_features.shape} vs {data.x[mask_nodes].shape}"

            # 计算逐样本的余弦相似度
            similarities = F.cosine_similarity(recon_features, data.x[mask_nodes], dim=-1)
            # 转换为损失：(1 - cos_sim)^gamma
            sample_loss = (1 - similarities).pow(self.gamma)
            recon_losses.append(sample_loss.mean())
        
        loss_input_recon = sum(recon_losses) / len(recon_losses)

        # --- 任务2: 潜在表征预测 ---
        # 教师模型处理未掩码数据
        with torch.no_grad():
            # 修正：调用新的 get_final_embeddings 接口，并只取我们需要的节点特征
            teacher_encoded_nodes, _ = self.teacher_encoder.get_final_embeddings(data)
        
        # MLP预测器预测
        predicted_latent = self.latent_pred_decoder(student_encoded_nodes)
        
        # 计算损失 (仅在掩码节点上)
        similarities_latent = F.cosine_similarity(
            predicted_latent[mask_nodes],
            teacher_encoded_nodes[mask_nodes].detach(),
            dim=-1
        )
        # 转换为损失：(1 - cos_sim)^gamma
        loss_latent_pred = (1 - similarities_latent).pow(self.gamma).mean()
        
        # --- 3. 计算总损失 (移除自适应权重) ---
        total_loss = loss_input_recon + self.latent_loss_lambda * loss_latent_pred

        # 添加数值稳定性检查
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"Warning: Invalid loss detected - total: {total_loss.item()}, recon: {loss_input_recon.item()}, latent: {loss_latent_pred.item()}")
            total_loss = torch.tensor(10.0, device=total_loss.device, requires_grad=True)  # fallback loss

        # 调试信息：报告相似度统计
        avg_input_sim = sum([1 - loss for loss in recon_losses]) / len(recon_losses) if recon_losses else 0.0
        avg_latent_sim = (1 - loss_latent_pred).item()

        return total_loss, {
            "loss_total": total_loss.item(),
            "L_input": loss_input_recon.item(),
            "L_latent": loss_latent_pred.item(),
            "adaptive_lambda": self.latent_loss_lambda, # 报告固定lambda值
            "debug_input_sim": avg_input_sim,
            "debug_latent_sim": avg_latent_sim,
            "debug_mask_rate": self.mask_rate,
            "debug_num_masked": mask_nodes.sum().item(),
            "debug_total_nodes": data.num_nodes
        }
