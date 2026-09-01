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
from src.neural_reconstruction.checkpoint import save_checkpoint, load_checkpoint

def run_experiment(args, is_smoke_test=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[{'SMOKE TEST' if is_smoke_test else 'FULL RUN'}] Using device: {device}")
    
    # 1. Configuration
    if is_smoke_test:
        resolution = (128, 72)
        batch_size = 512
        iterations = 100
        # Just use the first 10 frames for train, next 2 for val, next 3 for test
        train_indices = list(range(10))
        val_indices = list(range(10, 12))
        test_indices = list(range(12, 15))
    else:
        resolution = (256, 144)
        batch_size = 1024
        iterations = 2000
        # 350 frames total
        train_indices = list(range(280))
        val_indices = list(range(280, 315))
        test_indices = list(range(315, 350))
        
    out_dir = Path("outputs/reports/zurich_mav/b4")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Datasets
    print("Loading datasets...")
    train_ds = NeRFDataset(args.images, args.poses, args.intrinsics, args.image_dir, 'train', resolution, train_indices, device)
    val_ds = NeRFDataset(args.images, args.poses, args.intrinsics, args.image_dir, 'val', resolution, val_indices, device)
    test_ds = NeRFDataset(args.images, args.poses, args.intrinsics, args.image_dir, 'test', resolution, test_indices, device)
    
    # 3. Model & Trainer
    print("Initializing architecture...")
    model = TinyNeRF(hidden_dim=128, num_layers=6).to(device)
    renderer = VolumetricRenderer(bg_color=(0.0, 0.0, 0.0)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    
    near = 0.1
    far = 10.0
    trainer = Trainer(model, renderer, optimizer, device, near, far, num_samples=64)
    
    ckpt_path = out_dir / ("smoke_ckpt.pt" if is_smoke_test else "full_ckpt.pt")
    start_step, metrics = load_checkpoint(ckpt_path, model, optimizer)
    
    # 4. Training Loop
    print(f"Starting training from step {start_step} to {iterations}...")
    start_time = time.time()
    
    loss_history = metrics.get('loss_history', [])
    vram_history = metrics.get('vram_history', [])
    
    for step in range(start_step, iterations):
        rays_o, rays_d, target_rgb = train_ds.get_random_batch(batch_size)
        
        loss, _, _ = trainer.train_step(rays_o, rays_d, target_rgb)
        
        if step % 20 == 0 or step == iterations - 1:
            if torch.cuda.is_available():
                vram = torch.cuda.max_memory_allocated() / 1e9
                vram_history.append((step, vram))
            else:
                vram = 0.0
                
            loss_history.append((step, loss))
            print(f"Step {step:04d} | Loss: {loss:.4f} | VRAM: {vram:.2f} GB")
            
        if step % 500 == 0 and step > 0:
            save_checkpoint(ckpt_path, step, model, optimizer, {
                'loss_history': loss_history,
                'vram_history': vram_history
            })
            
    # Save final
    save_checkpoint(ckpt_path, iterations, model, optimizer, {
        'loss_history': loss_history,
        'vram_history': vram_history
    })
    
    # 5. Evaluation (Novel View)
    print("Evaluating novel views (validation set)...")
    val_loss = 0.0
    with torch.no_grad():
        rays_o, rays_d, gt_rgb = val_ds.get_image_rays(0)
        rendered_rgb, rendered_depth = trainer.render_image(rays_o, rays_d)
        val_loss = torch.nn.functional.mse_loss(rendered_rgb, gt_rgb).item()
        
        # Save validation image
        gt_img = (gt_rgb.cpu().numpy() * 255).astype(np.uint8)
        pred_img = (rendered_rgb.cpu().numpy() * 255).astype(np.uint8)
        
        # Convert RGB to BGR for OpenCV
        cv2.imwrite(str(out_dir / f"{'smoke' if is_smoke_test else 'full'}_val_gt.png"), cv2.cvtColor(gt_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_dir / f"{'smoke' if is_smoke_test else 'full'}_val_pred.png"), cv2.cvtColor(pred_img, cv2.COLOR_RGB2BGR))
        
        # Save depth map (normalize to 0-255)
        depth_np = rendered_depth.squeeze(-1).cpu().numpy()
        depth_norm = np.clip(depth_np / far * 255, 0, 255).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
        cv2.imwrite(str(out_dir / f"{'smoke' if is_smoke_test else 'full'}_val_depth.png"), depth_colored)
        
    runtime = time.time() - start_time
    
    # 6. Diagnostics output
    report = {
        'type': 'Smoke Test' if is_smoke_test else 'Full Run',
        'iterations': iterations,
        'batch_size': batch_size,
        'resolution': resolution,
        'final_train_loss': loss_history[-1][1] if loss_history else None,
        'novel_view_val_loss': val_loss,
        'peak_vram_gb': max([v for _, v in vram_history]) if vram_history else 0.0,
        'runtime_seconds': runtime,
        'status': 'PASS'
    }
    
    report_file = out_dir / ("b4_smoke_diagnostics.json" if is_smoke_test else "b4_training_metrics.json")
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=4)
        
    print(f"[{'SMOKE TEST' if is_smoke_test else 'FULL RUN'}] Finished. Peak VRAM: {report['peak_vram_gb']:.2f} GB")
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=Path, default=Path("outputs/reports/zurich_mav/images.csv"))
    parser.add_argument('--poses', type=Path, default=Path("outputs/reports/zurich_mav/b2/b2_fused_trajectory.csv"))
    parser.add_argument('--intrinsics', type=Path, default=Path("outputs/reports/zurich_mav/b0/reconstruction_summary.json"))
    parser.add_argument('--image_dir', type=Path, default=Path("D:/SIH26158/datasets/zurich_mav/AGZ_subset/MAV Images"))
    parser.add_argument('--full', action='store_true', help="Run the full experiment instead of smoke test")
    
    args = parser.parse_args()
    
    if not args.full:
        run_experiment(args, is_smoke_test=True)
    else:
        run_experiment(args, is_smoke_test=False)
