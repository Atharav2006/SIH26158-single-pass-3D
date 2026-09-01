import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, in_dim: int, num_freqs: int, include_input: bool = True):
        super().__init__()
        self.in_dim = in_dim
        self.num_freqs = num_freqs
        self.include_input = include_input
        self.out_dim = in_dim * (1 if include_input else 0) + in_dim * num_freqs * 2
        
        # Precompute frequencies (2^0, 2^1, ..., 2^(L-1)) * pi
        self.register_buffer(
            'freq_bands', 
            torch.pow(2.0, torch.arange(num_freqs, dtype=torch.float32)) * torch.pi
        )
        
    def forward(self, x):
        # x: [..., in_dim]
        out = [x] if self.include_input else []
        
        for freq in self.freq_bands:
            out.append(torch.sin(x * freq))
            out.append(torch.cos(x * freq))
            
        return torch.cat(out, dim=-1)

class TinyNeRF(nn.Module):
    def __init__(self, 
                 pos_in_dim: int = 3, 
                 dir_in_dim: int = 3, 
                 pos_freqs: int = 10, 
                 dir_freqs: int = 4, 
                 hidden_dim: int = 128, 
                 num_layers: int = 4):
        super().__init__()
        
        self.encode_pos = PositionalEncoding(pos_in_dim, pos_freqs)
        self.encode_dir = PositionalEncoding(dir_in_dim, dir_freqs)
        
        # Trunk for density
        layers = []
        in_dim = self.encode_pos.out_dim
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
        self.trunk = nn.Sequential(*layers)
        
        self.density_out = nn.Linear(hidden_dim, 1)
        # Initialize density bias to a small positive value to ensure non-zero
        # density output through F.relu at initialization. Without this, certain
        # random seeds produce all-negative pre-activation density, causing a
        # dead-ReLU zero-gradient trap where the model cannot learn.
        nn.init.constant_(self.density_out.bias, 0.1)
        
        # Head for RGB
        self.rgb_head = nn.Sequential(
            nn.Linear(hidden_dim + self.encode_dir.out_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 3),
            nn.Sigmoid()
        )
        
    def forward(self, x, d):
        """
        x: [..., 3] 3D coordinates
        d: [..., 3] Viewing directions (normalized)
        """
        batch_shape = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
        d = d.reshape(-1, d.shape[-1])
        
        pos_enc = self.encode_pos(x)
        dir_enc = self.encode_dir(d)
        
        features = self.trunk(pos_enc)
        
        # Density must be positive
        density = F.relu(self.density_out(features))
        
        # View-dependent color
        color = self.rgb_head(torch.cat([features, dir_enc], dim=-1))
        
        # Reshape back
        density = density.reshape(*batch_shape, 1)
        color = color.reshape(*batch_shape, 3)
        
        return density, color
