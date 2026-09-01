import argparse
import time
import json
from pathlib import Path
import torch
import torch.optim as optim
import numpy as np
import cv2

from src.neural_reconstruction.dataset import NeRFDataset
from src.neural_reconstruction.model import TinyNeRF
from src.neural_reconstruction.renderer import VolumetricRenderer
from src.neural_reconstruction.trainer import Trainer
from src.neural_reconstruction.depth_prior import MiDaSDepthPrior

def run_experiment_config(name, config, train_ds, val_ds, device, iterations, out_dir):
    print(f"\n=========================================")
    print(f"Starting Experiment: {name}")
    print(f"Config: {config}")
    print(f"=========================================")
    
    # Deterministic Init
    torch.manual_seed(42)
    np.random.seed(42)
    
    model = TinyNeRF(hidden_dim=128, num_layers=6).to(device)
    renderer = VolumetricRenderer(bg_color=(0.0, 0.0, 0.0)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    
    near, far = 0.1, 10.0
    trainer = Trainer(
        model, renderer, optimizer, device, near, far, num_samples=64,
        lambda_depth=config['lambda_depth'],
        lambda_smooth=config['lambda_smooth'],
        lambda_reg=config['lambda_reg']
    )
    
    start_time = time.time()
    loss_history = []
    vram_history = []
    
    for step in range(iterations):
        if config['lambda_depth'] > 0:
            # We must use single-image batching for scale-invariant depth loss
            rays_o, rays_d, target_rgb, target_depth = train_ds.get_single_image_batch(1024)
        else:
            rays_o, rays_d, target_rgb = train_ds.get_random_batch(1024)
            target_depth = None
            
        total_loss, l_rgb, l_depth, comp_rgb, comp_depth = trainer.train_step(rays_o, rays_d, target_rgb, target_depth)
        
        if step % 20 == 0 or step == iterations - 1:
            if torch.cuda.is_available():
                vram = torch.cuda.max_memory_allocated() / 1e9
                vram_history.append((step, vram))
            else:
                vram = 0.0
            loss_history.append((step, total_loss))
            print(f"[{name}] Step {step:04d} | Tot: {total_loss:.4f} | RGB: {l_rgb:.4f} | D: {l_depth:.4f} | VRAM: {vram:.2f} GB")

    # Evaluation
    print(f"[{name}] Evaluating novel views...")
    with torch.no_grad():
        rays_o, rays_d, gt_rgb = val_ds.get_image_rays(0)
        rendered_rgb, rendered_depth = trainer.render_image(rays_o, rays_d)
        val_loss = torch.nn.functional.mse_loss(rendered_rgb, gt_rgb).item()
        
        # Save validation image
        gt_img = (gt_rgb.cpu().numpy() * 255).astype(np.uint8)
        pred_img = (rendered_rgb.cpu().numpy() * 255).astype(np.uint8)
        
        cv2.imwrite(str(out_dir / f"{name}_val_gt.png"), cv2.cvtColor(gt_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_dir / f"{name}_val_pred.png"), cv2.cvtColor(pred_img, cv2.COLOR_RGB2BGR))
        
        # Save depth map
        depth_np = rendered_depth.squeeze(-1).cpu().numpy()
        depth_norm = np.clip(depth_np / far * 255, 0, 255).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
        cv2.imwrite(str(out_dir / f"{name}_val_depth.png"), depth_colored)
        
    runtime = time.time() - start_time
    
    return {
        'name': name,
        'config': config,
        'final_total_loss': loss_history[-1][1],
        'novel_view_val_loss': val_loss,
        'peak_vram_gb': max([v for _, v in vram_history]) if vram_history else 0.0,
        'runtime_seconds': runtime
    }

def run_all(args, is_smoke_test=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path("outputs/reports/zurich_mav/b4b")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if is_smoke_test:
        resolution = (128, 72)
        iterations = 100
        train_indices = list(range(5))
        val_indices = list(range(5, 7))
    else:
        resolution = (256, 144)
        iterations = 2000
        train_indices = list(range(280))
        val_indices = list(range(280, 315))
        
    print("Loading Depth Prior...")
    depth_prior = MiDaSDepthPrior(device)
    
    print("Loading datasets...")
    train_ds = NeRFDataset(args.images, args.poses, args.intrinsics, args.image_dir, 'train', resolution, train_indices, device, depth_prior=depth_prior)
    val_ds = NeRFDataset(args.images, args.poses, args.intrinsics, args.image_dir, 'val', resolution, val_indices, device, depth_prior=depth_prior)
    
    experiments = [
        {"name": "B4", "config": {"lambda_depth": 0.0, "lambda_smooth": 0.0, "lambda_reg": 0.0}},
        {"name": "B4_B", "config": {"lambda_depth": 1.0, "lambda_smooth": 0.0, "lambda_reg": 0.0}},
        {"name": "B4_B_Plus", "config": {"lambda_depth": 1.0, "lambda_smooth": 0.1, "lambda_reg": 0.01}}
    ]
    
    results = []
    for exp in experiments:
        res = run_experiment_config(
            exp["name"], exp["config"], 
            train_ds, val_ds, device, iterations, out_dir
        )
        results.append(res)
        
    report_file = out_dir / ("b4b_smoke_diagnostics.json" if is_smoke_test else "b4b_experiment_comparison.json")
    with open(report_file, 'w') as f:
        json.dump(results, f, indent=4)
        
    print("All experiments complete. Output saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=Path, default=Path("outputs/reports/zurich_mav/images.csv"))
    parser.add_argument('--poses', type=Path, default=Path("outputs/reports/zurich_mav/b2/b2_fused_trajectory.csv"))
    parser.add_argument('--intrinsics', type=Path, default=Path("outputs/reports/zurich_mav/b0/reconstruction_summary.json"))
    parser.add_argument('--image_dir', type=Path, default=Path("D:/SIH26158/datasets/zurich_mav/AGZ_subset/MAV Images"))
    parser.add_argument('--full', action='store_true')
    args = parser.parse_args()
    
    run_all(args, is_smoke_test=not args.full)
