"""
Stage 1: Dual SSL Pure Structure Pre-training
阶段一：双重自监督纯结构预训练

This script is the main entry point for the self-supervised pre-training
of the TEN-FMA framework's Module C. It implements the training loop for the
GraphMAE2 task, as defined in the GUIDELINES.
"""
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
import argparse
import os
import sys
from pathlib import Path
import json
import warnings
import time

# Ensure the src directory is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from src.simplex_data_loader import PerovskiteSimplexDataset
    from src.snn_model import SNN, SNNHamiltonianDynamicsSDE
    from src.ssl_tasks import GraphMAE2_SSL
except ImportError as e:
    print(f"Error: Failed to import necessary modules. Details: {e}")
    sys.exit(1)

def main(args):
    """Main training function."""
    
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 0. Set up Output Directory ---
    if args.model_id:
        timestamp = time.strftime("%m%d%H%M")
        output_dir = Path(args.output_dir) / f"model-{args.model_id}-{timestamp}"
    else:
        # Fallback for when no model_id is provided
        output_dir = Path(args.output_dir) / "pretrained_graphmae2_unnamed"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"模型输出路径设置为: {output_dir}")

    # --- 1. Data Loading ---
    print("Loading dataset...")
    try:
        dataset = PerovskiteSimplexDataset(
            data_root=args.data_root, 
            load_triangles=True,
            include_materials=args.include_materials
        )
        # For this test, we use a small subset. In real pre-training, use the full dataset.
        train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        stats = dataset.get_feature_stats()
        print(f"Dataset loaded successfully with {len(dataset)} samples.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # --- 2. Model Initialization ---
    print("Initializing models...")

    # 移除E3SNN逻辑，现在只支持标准的SNN模型
    snn_model = SNN(
        node_input_dim=stats['num_node_features'],
        edge_input_dim=stats['num_edge_features'],
        triangle_input_dim=stats.get('num_triangle_features', 0),
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        num_layers=args.num_layers
    )
    # 将SNN模型包装到SDE动态模块中
    encoder = SNNHamiltonianDynamicsSDE(
        snn_model=snn_model,
        integration_steps=args.sde_steps,
        temperature=args.temperature,
        friction=args.friction
    )
    print("Initialized standard SNN model with SDE dynamics.")


    ssl_model = GraphMAE2_SSL(
        student_encoder=encoder,
        mask_rate=args.mask_rate,
        remask_rate=args.remask_rate,
        num_remask_views=args.num_remask_views,
        latent_loss_lambda=args.lambda_latent,
        ema_decay=args.ema_decay,
        gamma=args.gamma  # 传入gamma参数
    ).to(device)
    
    # --- 3. Optimizer and Scheduler ---
    optimizer = optim.AdamW(ssl_model.student_encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 修正FutureWarning: 使用新的 torch.amp API
    # PyTorch 1.12+ 推荐使用 torch.amp 模块
    scaler = torch.amp.GradScaler('cuda', enabled=args.use_amp)

    # --- 4. Training Loop ---
    print("Starting pre-training...")
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        ssl_model.train()
        total_epoch_loss = 0
        
        for i, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            
            # 修正FutureWarning: 使用新的 torch.amp API
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=args.use_amp):
                total_loss, loss_dict = ssl_model(data)
            
            # Backward pass
            if torch.isnan(total_loss):
                warnings.warn(f"NaN loss detected at epoch {epoch}, step {i}. Skipping update.")
                continue

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Update teacher model with EMA
            ssl_model.update_teacher()

            total_epoch_loss += total_loss.item()
            
            if (i + 1) % args.log_interval == 0:
                # 修正KeyError: 使用正确的字典键名，并添加调试信息
                print(f"Epoch [{epoch+1}/{args.epochs}], Step [{i+1}/{len(train_loader)}], "
                      f"Loss: {total_loss.item():.4f}, "
                      f"L_input: {loss_dict['L_input']:.4f}, "
                      f"L_latent: {loss_dict['L_latent']:.4f}, "
                      f"Input_sim: {loss_dict.get('debug_input_sim', 0):.4f}, "
                      f"Latent_sim: {loss_dict.get('debug_latent_sim', 0):.4f}, "
                      f"Masked: {loss_dict.get('debug_num_masked', 0)}/{loss_dict.get('debug_total_nodes', 0)}")

        scheduler.step()
        
        avg_epoch_loss = total_epoch_loss / len(train_loader)
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        print(f"Epoch [{epoch+1}/{args.epochs}] completed in {epoch_duration:.2f}s. Average Loss: {avg_epoch_loss:.4f}")
    
    # --- 5. Save Model ---
    print("Pre-training finished. Saving model...")
    
    # Save only the student encoder's state dict, which is the final product
    torch.save(ssl_model.student_encoder.state_dict(), output_dir / "pretrained_encoder.pt")
    
    # Save hyperparameters
    with open(output_dir / "pretrain_args.json", 'w') as f:
        # 在保存的参数中，将output_dir更新为实际使用的路径
        args_dict = vars(args)
        args_dict['output_dir'] = str(output_dir)
        json.dump(args_dict, f, indent=4)
        
    print(f"Model and config saved to {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="GraphMAE2 Pre-training for TEN-FMA")

    # Data args
    parser.add_argument('--data_root', type=str, default='data', help='Root directory of the dataset.')
    parser.add_argument('--output_dir', type=str, default='models', help='Base directory to save the trained model.')
    parser.add_argument('--model_id', type=str, default=None, help='A unique identifier for the model, e.g., "4".')
    parser.add_argument('--include_materials', type=str, nargs='+', default=None, help='List of material formulas to include (e.g., CsPbBr3 CsPbI3).')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for data loading.')
   
    # Model args
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension size.')
    parser.add_argument('--output_dim', type=int, default=128, help='Output dimension of the SNN part of the encoder.')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of SNN layers.')

    # SDE args
    # 优化：根据用户反馈，将默认SDE积分步数降低以加速
    parser.add_argument('--sde_steps', type=int, default=5, help='Number of integration steps for the SDE solver.')
    parser.add_argument('--temperature', type=float, default=0.1, help='Temperature for Langevin dynamics.')
    parser.add_argument('--friction', type=float, default=0.01, help='Friction for Langevin dynamics.')

    # GraphMAE2 args
    parser.add_argument('--mask_rate', type=float, default=0.75, help='Rate of nodes to mask.')
    parser.add_argument('--remask_rate', type=float, default=0.5, help='Rate of encoded features to re-mask for the decoder.')
    parser.add_argument('--num_remask_views', type=int, default=4, help='Number of random re-masking views for the decoder.')
    parser.add_argument('--lambda_latent', type=float, default=1.0, help='Weight for the latent prediction loss. This is a crucial hyperparameter that needs to be tuned for different datasets (see GraphMAE2 paper).')
    parser.add_argument('--ema_decay', type=float, default=0.999, help='Decay rate for the EMA teacher model.')
    parser.add_argument('--gamma', type=float, default=2.0, help='Scaling factor gamma for the scaled cosine error loss.')

    # Training args
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size. Keep it small due to SDE memory usage.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for the optimizer.')
    parser.add_argument('--log_interval', type=int, default=1, help='Interval for logging training progress.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID to use.')
    parser.add_argument('--use_amp', action='store_true', default=False, help='Enable Automatic Mixed Precision (AMP) for training.')

    args = parser.parse_args()
    main(args)
