"""
SIH26158 Depth / AI Quality Diagnostics Analyzer (Member 3 TASK 8)

Calculates edge discontinuities and structural outliers from dense depth maps.
Safely marks AI metrics and multi-view metrics as NOT_AVAILABLE.
"""

import json
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional
import numpy as np


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DepthEdgeStatistics:
    """Discontinuity tracking across valid depth maps."""
    edge_energy_mean: float = 0.0
    discontinuity_ratio: float = 0.0
    availability: str = "NOT_AVAILABLE"


@dataclass
class DepthOutlierStatistics:
    """Statistical tracking of depth outliers via Median Absolute Deviation (MAD)."""
    outlier_count: int = 0
    outlier_ratio: float = 0.0
    mad: float = 0.0
    availability: str = "NOT_AVAILABLE"


@dataclass
class AIQualityStatistics:
    """Placeholder handles for AI models (NO MODEL PRESENT)."""
    model_name: str = "NOT_AVAILABLE"
    inference_health: str = "NOT_AVAILABLE"
    availability: str = "NOT_AVAILABLE"


@dataclass
class DepthAIQualityReport:
    """Consolidated report for advanced depth attributes."""
    depth_edges: Dict[str, Any]
    depth_outliers: Dict[str, Any]
    ai_quality: Dict[str, Any]
    quality: str  # GOOD | WARNING | POOR | FAILED
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 4) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class DepthAIAnalyzer:
    """
    Evaluates depth map robust statistics and safely handles non-available ML assumptions.
    """

    def analyze_depth_edges(
        self, depth_maps: List[np.ndarray], high_frequency_threshold: float = 0.5
    ) -> DepthEdgeStatistics:
        """
        Uses normalized Laplacian/gradient magnitude to assess discontinuity.
        """
        if not depth_maps:
            return DepthEdgeStatistics()

        valid_edge_scores = []
        total_valid_pixels = 0
        total_high_freq = 0

        for dmap in depth_maps:
            if dmap is None or dmap.size == 0 or dmap.ndim != 2:
                continue

            valid_mask = np.isfinite(dmap) & (dmap > 0.0)
            if not np.any(valid_mask):
                continue
                
            # Compute gradient magnitudes on valid depth maps padding 0s on boundaries
            dx = np.diff(dmap, axis=1)
            dy = np.diff(dmap, axis=0)
            
            # Simple average horizontal and vertical gradients
            dx_sq = dx**2
            dy_sq = dy**2
            
            # Pad to original shape
            grad_x = np.pad(dx_sq, ((0, 0), (0, 1)), mode='constant')
            grad_y = np.pad(dy_sq, ((0, 1), (0, 0)), mode='constant')
            
            grad_mag = np.sqrt(grad_x + grad_y)
            
            # Use only regions where both the pixel and its neighbor are valid
            valid_grad_mask = valid_mask.copy()
            valid_grad_mask[:, :-1] &= valid_mask[:, 1:]
            valid_grad_mask[:-1, :] &= valid_mask[1:, :]
            
            if np.any(valid_grad_mask):
                grad_vals = grad_mag[valid_grad_mask]
                
                # Normalize by depth value to get relative gradient (Weber fraction equivalent)
                rel_grad = grad_vals / dmap[valid_grad_mask]
                
                valid_edge_scores.append(rel_grad)
                total_valid_pixels += len(rel_grad)
                total_high_freq += int(np.sum(rel_grad > high_frequency_threshold))

        if not valid_edge_scores:
            return DepthEdgeStatistics(availability="AVAILABLE")

        all_edges = np.concatenate(valid_edge_scores)
        mean_energy = float(np.mean(all_edges))
        discontinuity_ratio = float(total_high_freq / total_valid_pixels)

        return DepthEdgeStatistics(
            edge_energy_mean=mean_energy,
            discontinuity_ratio=discontinuity_ratio,
            availability="AVAILABLE"
        )

    def analyze_depth_outliers(
        self, depth_maps: List[np.ndarray], mad_threshold: float = 3.0
    ) -> DepthOutlierStatistics:
        """
        Calculates robust Median Absolute Deviation (MAD) of depths to isolate outliers.
        """
        if not depth_maps:
            return DepthOutlierStatistics()

        all_valid_depths = []
        for dmap in depth_maps:
            if dmap is None or dmap.size == 0:
                continue
            valid_mask = np.isfinite(dmap) & (dmap > 0.0)
            if np.any(valid_mask):
                all_valid_depths.append(dmap[valid_mask])

        if not all_valid_depths:
            return DepthOutlierStatistics(availability="AVAILABLE")

        merged = np.concatenate(all_valid_depths)
        total = len(merged)

        if total == 0:
            return DepthOutlierStatistics(availability="AVAILABLE")

        median_val = np.median(merged)
        abs_dev = np.abs(merged - median_val)
        mad = float(np.median(abs_dev))

        if mad == 0:
            outlier_count = 0
            outlier_ratio = 0.0
        else:
            # Standard scale factor for normally distributed data is 1.4826
            # but relative depths don't fit perfectly; using simple raw unscaled deviation bounds.
            outliers = abs_dev > (mad_threshold * mad)
            outlier_count = int(np.sum(outliers))
            outlier_ratio = float(outlier_count / total)

        return DepthOutlierStatistics(
            outlier_count=outlier_count,
            outlier_ratio=outlier_ratio,
            mad=mad,
            availability="AVAILABLE"
        )

    def build_report(
        self,
        depth_edges: DepthEdgeStatistics,
        depth_outliers: DepthOutlierStatistics,
        ai_quality: AIQualityStatistics,
        discontinuity_warning_threshold: float = 0.6,
        outlier_warning_threshold: float = 0.15
    ) -> DepthAIQualityReport:
        """
        Combine metrics into a final diagnostic quality format.
        AI and Multi-view are guaranteed handled safely.
        """
        warnings = []
        recommendations = []
        is_poor = False
        is_failed = False

        if depth_edges.availability == "AVAILABLE":
            if depth_edges.discontinuity_ratio > discontinuity_warning_threshold:
                warnings.append(f"Highly discontinuous depth fields ({depth_edges.discontinuity_ratio*100:.1f}% edges).")
                recommendations.append("Consider filtering patch-match stereo with stronger smoothing.")

        if depth_outliers.availability == "AVAILABLE":
            if depth_outliers.outlier_ratio > outlier_warning_threshold:
                warnings.append(f"High localized depth outliers ({depth_outliers.outlier_ratio*100:.1f}%, MAD={depth_outliers.mad:.3f}).")
                is_poor = True
                recommendations.append("Filter outliers pre-fusion or lower depth threshold ranges.")

        if is_failed:
            quality = "FAILED"
        elif is_poor:
            quality = "POOR"
        elif warnings:
            quality = "WARNING"
        else:
            quality = "GOOD"

        return DepthAIQualityReport(
            depth_edges=asdict(depth_edges),
            depth_outliers=asdict(depth_outliers),
            ai_quality=asdict(ai_quality),
            quality=quality,
            warnings=warnings,
            recommendations=recommendations
        )


def generate_depth_ai_quality_report(report: DepthAIQualityReport) -> str:
    e = report.depth_edges
    o = report.depth_outliers
    a = report.ai_quality

    lines = [
        "============================================================",
        "          DEPTH / AI RECONSTRUCTION QUALITY REPORT",
        "============================================================",
        f"OVERALL QUALITY           : {report.quality}",
        "------------------------------------------------------------",
        "DEPTH STRUCTURAL CONTINUITY",
        f"Availability              : {e.get('availability', 'NOT_AVAILABLE')}",
    ]
    if e.get("availability") == "AVAILABLE":
        lines.extend([
            f"Mean Edge Energy (Rel)    : {e.get('edge_energy_mean', 0.0):.4f}",
            f"High-Freq Discontinuity   : {e.get('discontinuity_ratio', 0.0)*100:.2f}%",
        ])
    lines.extend([
        "------------------------------------------------------------",
        "DEPTH OUTLIER STATISTICS (MAD)",
        f"Availability              : {o.get('availability', 'NOT_AVAILABLE')}",
    ])
    if o.get("availability") == "AVAILABLE":
        lines.extend([
            f"Median Absolute Dev (MAD) : {o.get('mad', 0.0):.4f}",
            f"Outliers Count            : {o.get('outlier_count', 0):,}",
            f"Outlier Ratio             : {o.get('outlier_ratio', 0.0)*100:.2f}%",
        ])
    lines.extend([
        "------------------------------------------------------------",
        "NEURAL / AI MODEL METRICS",
        f"Availability              : {a.get('availability', 'NOT_AVAILABLE')}",
    ])
    if a.get("availability") == "AVAILABLE":
        lines.extend([
            f"Model Identity            : {a.get('model_name', 'NOT_AVAILABLE')}",
            f"Inference Health          : {a.get('inference_health', 'NOT_AVAILABLE')}",
        ])
    lines.extend([
        "Multi-View Consistency    : NOT_AVAILABLE (Strict Relative Geometry)",
        "Temporal Consistency      : NOT_AVAILABLE (Unordered Sequence)",
        "------------------------------------------------------------",
    ])

    if report.warnings:
        lines.append("WARNINGS:")
        for w in report.warnings:
            lines.append(f"  [!] {w}")
        lines.append("------------------------------------------------------------")

    if report.recommendations:
        lines.append("RECOMMENDATIONS:")
        for rec in report.recommendations:
            lines.append(f"  -> {rec}")
        lines.append("------------------------------------------------------------")

    return "\n".join(lines)
