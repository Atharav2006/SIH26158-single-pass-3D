from dataclasses import dataclass
import torch
import torch.nn.functional as F
from typing import Dict, Any, Union

@dataclass
class DepthPrediction:
    depth: torch.Tensor
    confidence: torch.Tensor
    source_type: str
    scale_type: str
    metadata: Dict[str, Any]

class DepthPrior:
    """Reusable interface for depth priors."""
    def __init__(self, device: torch.device):
        self.device = device

    def predict(self, image: torch.Tensor) -> DepthPrediction:
        raise NotImplementedError

    def predict_batch(self, images: torch.Tensor) -> DepthPrediction:
        raise NotImplementedError

    def estimate_uncertainty(self, image: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def metadata(self) -> Dict[str, Any]:
        raise NotImplementedError

class MiDaSDepthPrior(DepthPrior):
    def __init__(self, device: torch.device):
        super().__init__(device)
        self.source_type = "MiDaS_small"
        self.scale_type = "relative_inverse_depth"
        self._load_model()

    def _load_model(self):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Workaround for torch hub efficientnet
            try:
                torch.hub.load("rwightman/gen-efficientnet-pytorch", "gen_efficientnet_lite0", trust_repo=True, pretrained=False)
            except Exception:
                pass
            
            self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
            self.model.to(self.device)
            self.model.eval()

    @torch.no_grad()
    def predict_batch(self, images: torch.Tensor) -> DepthPrediction:
        """
        images: [B, H, W, 3] RGB tensor in [0, 1]
        Returns DepthPrediction
        """
        B, H, W, _ = images.shape
        
        img_pt = images.permute(0, 3, 1, 2)
        img_pt = F.interpolate(img_pt, size=(256, 256), mode='bilinear', align_corners=False)
        
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        img_pt = (img_pt - mean) / std
        
        pred_inv_depth = self.model(img_pt) # [B, 256, 256]
        
        pred_inv_depth = F.interpolate(
            pred_inv_depth.unsqueeze(1),
            size=(H, W),
            mode="bicubic",
            align_corners=False,
        ).squeeze(1)
        
        # MiDaS does not naturally predict confidence. Defaulting to 1.0.
        confidence = torch.ones_like(pred_inv_depth)
        
        return DepthPrediction(
            depth=pred_inv_depth,
            confidence=confidence,
            source_type=self.source_type,
            scale_type=self.scale_type,
            metadata={"resolution_inference": (256, 256), "normalization": "ImageNet"}
        )

    def predict(self, image: torch.Tensor) -> DepthPrediction:
        """image: [H, W, 3] RGB tensor"""
        res = self.predict_batch(image.unsqueeze(0))
        return DepthPrediction(
            depth=res.depth.squeeze(0),
            confidence=res.confidence.squeeze(0),
            source_type=res.source_type,
            scale_type=res.scale_type,
            metadata=res.metadata
        )

    def estimate_uncertainty(self, image: torch.Tensor) -> torch.Tensor:
        # Standard MiDaS lacks uncertainty. Returning flat 0 (no uncertainty) or flat 1 depending on spec.
        # We'll return 0 (confident) for now, but this interface supports bayesian models later.
        return torch.zeros(image.shape[:2], device=self.device)

    def metadata(self) -> Dict[str, Any]:
        return {
            "source_type": self.source_type,
            "scale_type": self.scale_type,
            "metric": False
        }
