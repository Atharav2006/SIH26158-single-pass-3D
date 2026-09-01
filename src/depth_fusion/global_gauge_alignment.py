import numpy as np
import warnings
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.sparse.csgraph import connected_components
from scipy.optimize import least_squares
from typing import List, Dict, Tuple, Optional, Any
from enum import Enum
import cv2

class GaugeRepresentation(Enum):
    D_INV = "D_inv"
    D_REL = "D_rel"
    
class GaugeAlignmentStatus(Enum):
    RELATIVE_GAUGE_ESTABLISHED = "RELATIVE_GAUGE_ESTABLISHED"
    RELATIVE_GAUGE_PARTIALLY_ESTABLISHED = "RELATIVE_GAUGE_PARTIALLY_ESTABLISHED"
    RELATIVE_GAUGE_NOT_IDENTIFIABLE = "RELATIVE_GAUGE_NOT_IDENTIFIABLE"
    REPRESENTATION_SELECTION_AMBIGUOUS = "REPRESENTATION_SELECTION_AMBIGUOUS"

def pose_aware_correspondences(
    depth_i: np.ndarray, 
    K_rect: np.ndarray, 
    R_wc_i: np.ndarray, 
    C_w_i: np.ndarray,
    depth_j: np.ndarray, 
    R_wc_j: np.ndarray, 
    C_w_j: np.ndarray,
    conf_i: Optional[np.ndarray] = None,
    conf_j: Optional[np.ndarray] = None,
    downsample_factor: int = 4
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Establishes pose-aware geometric correspondences between two frames.
    Returns: D_i_vals, D_j_vals, weights
    """
    H, W = depth_i.shape
    
    # Downsample for performance in gauge estimation
    y, x = np.mgrid[0:H:downsample_factor, 0:W:downsample_factor]
    u = x.flatten()
    v = y.flatten()
    
    Z_i = depth_i[v, u]
    valid_i = (Z_i > 0)
    if conf_i is not None:
        valid_i &= (conf_i[v, u] > 0.1)
        
    u, v, Z_i = u[valid_i], v[valid_i], Z_i[valid_i]
    
    # Ray generation
    fx, fy = K_rect[0, 0], K_rect[1, 1]
    cx, cy = K_rect[0, 2], K_rect[1, 2]
    
    X_c = (u - cx) * Z_i / fx
    Y_c = (v - cy) * Z_i / fy
    pts_c_i = np.stack([X_c, Y_c, Z_i], axis=-1)
    
    # To World
    pts_w = pts_c_i @ R_wc_i.T + C_w_i.reshape(1, 3)
    
    # To Camera j
    R_cw_j = R_wc_j.T
    pts_c_j = (pts_w - C_w_j.reshape(1, 3)) @ R_cw_j.T
    
    # Project to pixels j
    Z_c_j = pts_c_j[:, 2]
    valid_z = Z_c_j > 1e-6
    pts_c_j = pts_c_j[valid_z]
    u_i, v_i = u[valid_z], v[valid_z]
    Z_i_valid = Z_i[valid_z]
    
    u_j = (pts_c_j[:, 0] * fx / pts_c_j[:, 2]) + cx
    v_j = (pts_c_j[:, 1] * fy / pts_c_j[:, 2]) + cy
    
    u_j_int = np.round(u_j).astype(int)
    v_j_int = np.round(v_j).astype(int)
    
    # Bounds check
    in_bounds = (u_j_int >= 0) & (u_j_int < W) & (v_j_int >= 0) & (v_j_int < H)
    u_j_int, v_j_int = u_j_int[in_bounds], v_j_int[in_bounds]
    Z_i_valid = Z_i_valid[in_bounds]
    u_i, v_i = u_i[in_bounds], v_i[in_bounds]
    
    # Sample D_j
    D_j_vals = depth_j[v_j_int, u_j_int]
    valid_j = D_j_vals > 0
    if conf_j is not None:
        valid_j &= (conf_j[v_j_int, u_j_int] > 0.1)
        
    D_j_vals = D_j_vals[valid_j]
    D_i_vals = Z_i_valid[valid_j]
    
    # Weight combining confidences
    weights = np.ones_like(D_i_vals)
    if conf_i is not None and conf_j is not None:
        c_i = conf_i[v_i[valid_j], u_i[valid_j]]
        c_j = conf_j[v_j_int[valid_j], u_j_int[valid_j]]
        weights = c_i * c_j
        
    return D_i_vals, D_j_vals, weights

def fit_pairwise_affine(D_i: np.ndarray, D_j: np.ndarray, weights: np.ndarray) -> Dict[str, Any]:
    """Fits D_j ~ a*D_i + b robustly."""
    if len(D_i) < 100:
        return {"status": "INSUFFICIENT_POINTS"}
        
    # Variance check
    if np.var(D_i) < 1e-6 or np.var(D_j) < 1e-6:
        return {"status": "ZERO_VARIANCE"}
        
    # Condition number
    A_mat = np.vstack([D_i, np.ones_like(D_i)]).T
    cond_num = np.linalg.cond(A_mat)
    if cond_num > 1e6:
        return {"status": "POOR_CONDITIONING"}
        
    # Robust Huber Fit
    def res(p): return (p[0]*D_i + p[1] - D_j) * np.sqrt(weights)
    
    # Initial guess
    a_guess = np.std(D_j) / np.std(D_i)
    b_guess = np.mean(D_j) - a_guess * np.mean(D_i)
    
    opt = least_squares(res, [a_guess, b_guess], loss='huber', f_scale=np.median(np.abs(D_j - np.median(D_j)))*0.5 + 1e-3)
    a_fit, b_fit = opt.x
    
    if a_fit <= 0:
        return {"status": "NEGATIVE_SCALE"}
        
    pred = a_fit * D_i + b_fit
    residuals = np.abs(pred - D_j)
    mae = np.mean(residuals)
    norm_rmse = np.sqrt(np.mean(residuals**2)) / (np.std(D_j) + 1e-6)
    
    r = np.corrcoef(D_i, D_j)[0, 1]
    
    return {
        "status": "SUCCESS",
        "a": a_fit,
        "b": b_fit,
        "correlation": r,
        "residual": mae,
        "norm_rmse": norm_rmse,
        "cond": cond_num,
        "valid_count": len(D_i)
    }

class GlobalGaugeSolver:
    """
    Primary solver for Global Relative-Depth Gauge Alignment.
    Enforces graph consistency: a_j * a_ij = a_i and b_i = a_j * b_ij + b_j
    using a decoupled, unregularized least-squares optimization.
    """
    def __init__(self, ref_frame: int = 0):
        self.ref_frame = ref_frame
        
    def check_degeneracy(self, edges: List[Dict], num_frames: int) -> Optional[str]:
        if num_frames == 0: return "ZERO_FRAMES"
        if len(edges) == 0: return "NO_EDGES"
            
        adj = sp.lil_matrix((num_frames, num_frames))
        for e in edges:
            adj[e['i'], e['j']] = 1
            adj[e['j'], e['i']] = 1
            
        n_components = connected_components(csgraph=adj, directed=False, return_labels=False)
        if n_components > 1:
            return "DISCONNECTED_GRAPH"
            
        for e in edges:
            if e['a'] <= 0:
                return "INVALID_AFFINE_PARAMETER"
                
        return None

    def solve(self, edges: List[Dict], num_frames: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
        degeneracy = self.check_degeneracy(edges, num_frames)
        if degeneracy:
            return None, None, degeneracy
            
        num_edges = len(edges)
        A_scale = sp.lil_matrix((num_edges, num_frames))
        B_scale = np.zeros(num_edges)
        W = np.zeros(num_edges)
        
        for k, e in enumerate(edges):
            i, j, a_ij, w = e['i'], e['j'], e['a'], e.get('w', 1.0)
            A_scale[k, i] = 1.0
            A_scale[k, j] = -1.0
            B_scale[k] = np.log(a_ij)
            W[k] = w
            
        mask = np.ones(num_frames, dtype=bool)
        mask[self.ref_frame] = False
        
        A_reduced = A_scale.tocsr()[:, mask]
        W_mat = sp.diags(W)
        AtW = A_reduced.T.dot(W_mat)
        
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                x_reduced = spsolve(AtW.dot(A_reduced), AtW.dot(B_scale))
                if len(w) > 0 and issubclass(w[-1].category, sp.linalg.MatrixRankWarning):
                    return None, None, "RANK_DEFICIENCY"
        except RuntimeError: 
            return None, None, "RANK_DEFICIENCY"
            
        x = np.zeros(num_frames)
        x[mask] = x_reduced
        a_global = np.exp(x)
        
        A_shift = sp.lil_matrix((num_edges, num_frames))
        B_shift = np.zeros(num_edges)
        
        for k, e in enumerate(edges):
            i, j, b_ij, w = e['i'], e['j'], e['b'], e.get('w', 1.0)
            A_shift[k, i] = 1.0
            A_shift[k, j] = -1.0
            B_shift[k] = a_global[j] * b_ij
            
        A_shift_reduced = A_shift.tocsr()[:, mask]
        AtW_shift = A_shift_reduced.T.dot(W_mat)
        
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                y_reduced = spsolve(AtW_shift.dot(A_shift_reduced), AtW_shift.dot(B_shift))
                if len(w) > 0 and issubclass(w[-1].category, sp.linalg.MatrixRankWarning):
                    return None, None, "RANK_DEFICIENCY"
        except RuntimeError:
            return None, None, "RANK_DEFICIENCY"
            
        b_global = np.zeros(num_frames)
        b_global[mask] = y_reduced
        
        return a_global, b_global, "SUCCESS"

def evaluate_representations(edges_inv: List[Dict], edges_rel: List[Dict], num_frames: int) -> Tuple[GaugeRepresentation, Dict]:
    """
    Evaluates D_inv versus D_rel based on predefined criteria.
    Returns the selected representation and the metrics dict.
    """
    def compute_metrics(edges):
        if not edges: return {"composite_score": -np.inf}
        r_vals = [e['correlation'] for e in edges if 'correlation' in e]
        rmse_vals = [e['norm_rmse'] for e in edges if 'norm_rmse' in e]
        cond_vals = [e['cond'] for e in edges if 'cond' in e]
        count_vals = [e['valid_count'] for e in edges if 'valid_count' in e]
        
        mean_r = np.mean(r_vals) if r_vals else 0
        mean_rmse = np.mean(rmse_vals) if rmse_vals else np.inf
        mean_cond = np.mean(cond_vals) if cond_vals else np.inf
        mean_count = np.mean(count_vals) if count_vals else 0
        
        # Composite rule (heuristic weights to balance scale magnitude differences):
        # r is [0,1]. rmse is [0, inf). cond is [1, inf). 
        # Normalize terms roughly.
        score = (mean_r * 100.0) - (mean_rmse * 10.0) - np.log10(mean_cond + 1e-6) + np.log10(mean_count + 1e-6)
        return {
            "mean_correlation": float(mean_r),
            "mean_norm_rmse": float(mean_rmse),
            "mean_condition_number": float(mean_cond),
            "mean_valid_correspondences": float(mean_count),
            "composite_score": float(score)
        }
        
    m_inv = compute_metrics(edges_inv)
    m_rel = compute_metrics(edges_rel)
    
    if m_inv["composite_score"] == -np.inf and m_rel["composite_score"] == -np.inf:
        return None, {"D_inv": m_inv, "D_rel": m_rel, "selected": "NONE"}
        
    selected = GaugeRepresentation.D_INV if m_inv["composite_score"] >= m_rel["composite_score"] else GaugeRepresentation.D_REL
    
    # If the scores are indistinguishably close (e.g. < 0.1 diff), mark as ambiguous
    if abs(m_inv["composite_score"] - m_rel["composite_score"]) < 0.1:
        return None, {"D_inv": m_inv, "D_rel": m_rel, "selected": "AMBIGUOUS"}
        
    return selected, {"D_inv": m_inv, "D_rel": m_rel, "selected": selected.value}

def align_sequence(
    depth_sequence: Dict[int, Dict[str, Any]],
    overlap_graph: List[Tuple[int, int]],
    reference_frame: int = 0,
    representation: Optional[GaugeRepresentation] = None
) -> Dict[str, Any]:
    """
    Main entry point for Global Relative-Depth Gauge Alignment.
    
    depth_sequence: dict of frame_id -> {
        "D_inv": np.ndarray,
        "conf": Optional[np.ndarray],
        "K_rect": np.ndarray,
        "R_wc": np.ndarray,
        "C_w": np.ndarray
    }
    overlap_graph: list of frame_id pairs (i, j)
    """
    frame_ids = sorted(list(depth_sequence.keys()))
    if not frame_ids:
        return {"status": GaugeAlignmentStatus.RELATIVE_GAUGE_NOT_IDENTIFIABLE.value}
        
    # Map frame_ids to 0..N-1 indices for the solver
    id_to_idx = {fid: idx for idx, fid in enumerate(frame_ids)}
    num_frames = len(frame_ids)
    
    if reference_frame not in id_to_idx:
        reference_frame = frame_ids[0]
    ref_idx = id_to_idx[reference_frame]
    
    edges_inv = []
    edges_rel = []
    
    rejected_edges = []
    
    for (i_id, j_id) in overlap_graph:
        if i_id not in depth_sequence or j_id not in depth_sequence:
            continue
            
        data_i = depth_sequence[i_id]
        data_j = depth_sequence[j_id]
        
        # We assume D_inv is given
        D_i_inv = data_i["D_inv"]
        D_j_inv = data_j["D_inv"]
        
        D_i_vals, D_j_vals, weights = pose_aware_correspondences(
            D_i_inv, data_i["K_rect"], data_i["R_wc"], data_i["C_w"],
            D_j_inv, data_j["R_wc"], data_j["C_w"],
            conf_i=data_i.get("conf"), conf_j=data_j.get("conf")
        )
        
        # Fit D_inv
        fit_inv = fit_pairwise_affine(D_i_vals, D_j_vals, weights)
        if fit_inv["status"] == "SUCCESS":
            fit_inv.update({'i': id_to_idx[i_id], 'j': id_to_idx[j_id], 'w': fit_inv['valid_count']})
            edges_inv.append(fit_inv)
        else:
            rejected_edges.append({"edge": (i_id, j_id), "rep": "D_inv", "reason": fit_inv["status"]})
            
        # Fit D_rel
        epsilon = 1e-4
        D_i_rel_vals = 1.0 / (D_i_vals + epsilon)
        D_j_rel_vals = 1.0 / (D_j_vals + epsilon)
        
        fit_rel = fit_pairwise_affine(D_i_rel_vals, D_j_rel_vals, weights)
        if fit_rel["status"] == "SUCCESS":
            fit_rel.update({'i': id_to_idx[i_id], 'j': id_to_idx[j_id], 'w': fit_rel['valid_count']})
            edges_rel.append(fit_rel)
        else:
            rejected_edges.append({"edge": (i_id, j_id), "rep": "D_rel", "reason": fit_rel["status"]})
            
    # Selection
    metrics_info = {}
    if representation is None:
        selected_rep, metrics_info = evaluate_representations(edges_inv, edges_rel, num_frames)
        if selected_rep is None:
            return {
                "status": GaugeAlignmentStatus.REPRESENTATION_SELECTION_AMBIGUOUS.value,
                "metrics": metrics_info
            }
        representation = selected_rep
    
    active_edges = edges_inv if representation == GaugeRepresentation.D_INV else edges_rel
    
    solver = GlobalGaugeSolver(ref_frame=ref_idx)
    a_global, b_global, solver_status = solver.solve(active_edges, num_frames)
    
    if solver_status != "SUCCESS":
        return {
            "status": GaugeAlignmentStatus.RELATIVE_GAUGE_NOT_IDENTIFIABLE.value,
            "reason": solver_status,
            "metrics": metrics_info
        }
        
    # Build aligned depth sequence
    aligned_depths = {}
    global_scales = {}
    global_shifts = {}
    
    for fid, idx in id_to_idx.items():
        D = depth_sequence[fid]["D_inv"]
        if representation == GaugeRepresentation.D_REL:
            D = 1.0 / (D + epsilon)
            
        a = a_global[idx]
        b = b_global[idx]
        
        D_aligned = a * D + b
        
        aligned_depths[fid] = D_aligned
        global_scales[fid] = a
        global_shifts[fid] = b
        
    # Before/After drift/residual computation
    mean_res_before = np.mean([e['residual'] for e in active_edges]) if active_edges else 0
    # True residual after alignment: G_j(D_j) - G_i(D_i) => a_j D_j + b_j - (a_i D_i + b_i)
    # Using the local fit: D_j ≈ a_ij D_i + b_ij
    # Let's approximate global residual by how well the parameters fit
    # a_j a_ij \approx a_i -> error in scale
    scale_var_after = np.std([a_global[e['i']] - a_global[e['j']]*e['a'] for e in active_edges]) if active_edges else 0
    shift_var_after = np.std([b_global[e['i']] - (a_global[e['j']]*e['b'] + b_global[e['j']]) for e in active_edges]) if active_edges else 0
    
    return {
        "status": GaugeAlignmentStatus.RELATIVE_GAUGE_ESTABLISHED.value,
        "aligned_depths": aligned_depths,
        "global_scales": global_scales,
        "global_shifts": global_shifts,
        "representation": representation.value,
        "metrics": metrics_info,
        "diagnostics": {
            "reference_frame": reference_frame,
            "num_frames": num_frames,
            "num_edges": len(active_edges),
            "rejected_edges": rejected_edges,
            "mean_residual_before": mean_res_before,
            "scale_consistency_error": scale_var_after,
            "shift_consistency_error": shift_var_after
        },
        "edges_used": active_edges
    }
