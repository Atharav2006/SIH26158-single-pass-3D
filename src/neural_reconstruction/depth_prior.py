import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

class DepthPrior:
    """
    Interface for extracting monocular relative depth priors.
    Maintains a clean separation between the neural renderer and the depth network.
    """
    def __init__(self, device: torch.device):
        self.device = device
        self.model = None
        self.transform = None
        self.source_type = "UNINITIALIZED"

    def _load_model(self):
        raise NotImplementedError

    def predict(self, image: torch.Tensor) -> torch.Tensor:
        """
        image: [H, W, 3] RGB tensor in [0, 1]
        Returns: [H, W] relative depth map (or disparity-like map)
        """
        raise NotImplementedError

    def predict_batch(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: [B, H, W, 3] RGB tensor in [0, 1]
        Returns: [B, H, W] relative depth maps
        """
        raise NotImplementedError

    def normalize_depth(self, depth: torch.Tensor) -> torch.Tensor:
        """
        Since monocular depth is scale/shift ambiguous, 
        we zero-mean and unit-variance normalize it to allow alignment.
        """
        mean = depth.mean(dim=[-1, -2], keepdim=True)
        std = depth.std(dim=[-1, -2], keepdim=True) + 1e-8
        return (depth - mean) / std
        
    def confidence(self, depth: torch.Tensor) -> torch.Tensor:
        """
        Returns a confidence mask [0, 1]. For MiDaS, edges are often unreliable.
        We provide a flat 1.0 confidence mask by default unless overridden.
        """
        return torch.ones_like(depth)

    def metadata(self):
        return {
            "source_type": self.source_type,
            "metric": False, # MiDaS produces relative disparity/depth
            "scale_invariant": True
        }

class MiDaSDepthPrior(DepthPrior):
    def __init__(self, device: torch.device):
        super().__init__(device)
        self.source_type = "MiDaS_small (Relative Inverse Depth)"
        self._load_model()

    def _load_model(self):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Workaround to trust the underlying efficientnet repo
            try:
                torch.hub.load("rwightman/gen-efficientnet-pytorch", "gen_efficientnet_lite0", trust_repo=True, pretrained=False)
            except Exception:
                pass
            
            # Load MiDaS Small to respect VRAM limits
            self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
            self.model.to(self.device)
            self.model.eval()
            
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
            self.transform = midas_transforms.small_transform

    @torch.no_grad()
    def predict_batch(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: [B, H, W, 3] torch tensor in [0, 1]
        """
        B, H, W, _ = images.shape
        
        # Convert torch [B, H, W, 3] float -> numpy uint8 for MiDaS transform
        # (MiDaS transform expects numpy arrays)
        # However, calling it inside training loop repeatedly is slow.
        # Let's do it directly in PyTorch.
        # MiDaS small expects [B, 3, 256, 256] roughly, normalized by ImageNet stats.
        
        # Fast PyTorch-only transform for MiDaS small
        img_pt = images.permute(0, 3, 1, 2) # [B, 3, H, W]
        img_pt = F.interpolate(img_pt, size=(256, 256), mode='bilinear', align_corners=False)
        
        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        img_pt = (img_pt - mean) / std
        
        prediction = self.model(img_pt) # [B, 256, 256]
        
        # Resize back to original
        prediction = F.interpolate(
            prediction.unsqueeze(1),
            size=(H, W),
            mode="bicubic",
            align_corners=False,
        ).squeeze(1) # [B, H, W]
        
        return prediction
