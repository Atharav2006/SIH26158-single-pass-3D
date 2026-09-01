import torch

def compute_scale_and_shift(prediction, target, mask=None):
    """
    prediction: [B, ...] expected depth from NeRF
    target: [B, ...] pseudo-depth from MiDaS
    mask: [B, ...] boolean mask of valid pixels
    
    Solves for s, t such that s * prediction + t ≈ target
    Returns s, t
    """
    if mask is None:
        mask = torch.ones_like(prediction, dtype=torch.bool)
        
    p = prediction[mask]
    t = target[mask]
    
    # We want s * p + t - t = 0
    # Let's solve using least squares:
    # [p, 1] [s; t] = [t]
    
    # Simple zero-mean unit-variance alignment:
    # Instead of full least squares which can be unstable if variance is 0,
    # we can align by mean and std.
    
    var_p, mean_p = torch.var_mean(p)
    var_t, mean_t = torch.var_mean(t)
    
    # Protect against div by zero
    std_p = torch.sqrt(var_p + 1e-8)
    std_t = torch.sqrt(var_t + 1e-8)
    
    s = std_t / std_p
    t_shift = mean_t - s * mean_p
    
    return s, t_shift

def scale_invariant_depth_loss(prediction, target, mask=None):
    """
    Computes scale and shift invariant depth loss.
    Returns a scale-free normalized MSE (bounded 0 to 1).
    """
    s, t_shift = compute_scale_and_shift(prediction, target, mask)
    aligned_prediction = s * prediction + t_shift
    
    if mask is not None:
        loss = torch.nn.functional.mse_loss(aligned_prediction[mask], target[mask])
        var_t = torch.var(target[mask])
    else:
        loss = torch.nn.functional.mse_loss(aligned_prediction, target)
        var_t = torch.var(target)
        
    # Normalize by target variance to keep loss bounded ~ [0, 1]
    return loss / (var_t + 1e-8)

def edge_aware_smoothness_loss(depth, image):
    """
    depth: [B, H, W, 1]
    image: [B, H, W, 3]
    """
    # Gradients of depth
    depth_dx = depth[:, :, 1:, :] - depth[:, :, :-1, :]
    depth_dy = depth[:, 1:, :, :] - depth[:, :-1, :, :]
    
    # Gradients of image
    image_dx = image[:, :, 1:, :] - image[:, :, :-1, :]
    image_dy = image[:, 1:, :, :] - image[:, :-1, :, :]
    
    # Edge-aware weights
    weight_dx = torch.exp(-torch.mean(torch.abs(image_dx), dim=-1, keepdim=True))
    weight_dy = torch.exp(-torch.mean(torch.abs(image_dy), dim=-1, keepdim=True))
    
    loss_dx = torch.mean(torch.abs(depth_dx) * weight_dx)
    loss_dy = torch.mean(torch.abs(depth_dy) * weight_dy)
    
    return loss_dx + loss_dy

def density_regularization(density):
    """
    Penalizes 'fog' (widespread low density).
    """
    return torch.mean(torch.log1p(density))
