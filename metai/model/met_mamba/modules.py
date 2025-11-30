# metai/model/met_mamba/modules.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from timm.layers.drop import DropPath
from timm.layers.weight_init import trunc_normal_

# 尝试导入 Mamba
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    class Mamba(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            raise ImportError("Please install 'mamba_ssm': pip install mamba-ssm")

# ==========================================
# 基础组件
# ==========================================

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return output * self.weight

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, 
                 dilation=1, upsampling=False, act_norm=False):
        super().__init__()
        self.act_norm = act_norm
        if upsampling:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels*4, kernel_size, stride, padding, dilation),
                nn.PixelShuffle(2)
            )
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation)
        
        self.norm = nn.GroupNorm(2, out_channels)
        self.act = nn.SiLU(inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.act(self.norm(y))
        return y

class ConvSC(nn.Module):
    def __init__(self, C_in, C_out, kernel_size=3, downsampling=False, upsampling=False, act_norm=True):
        super().__init__()
        stride = 2 if downsampling else 1
        padding = (kernel_size - stride + 1) // 2
        self.conv = BasicConv2d(C_in, C_out, kernel_size, stride, padding, upsampling=upsampling, act_norm=act_norm)

    def forward(self, x):
        return self.conv(x)

class ResizeConv(nn.Module):
    """
    使用 "双线性插值 + 卷积" 替代 PixelShuffle，消除棋盘格伪影。
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, act_norm=True):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.norm = nn.GroupNorm(2, out_channels) if act_norm else nn.Identity()
        self.act = nn.SiLU(inplace=True) if act_norm else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.act(self.norm(x))
        return x

class LocalityEnhancedMLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, x, H, W):
        # x: [B, L, C]
        x = self.fc1(x)
        B, L, C = x.shape
        T = L // (H * W)
        
        # Reshape for DWConv: [B*T, C, H, W]
        x_spatial = x.view(B * T, H, W, C).permute(0, 3, 1, 2)
        x_spatial = self.dwconv(x_spatial)
        x = x_spatial.permute(0, 2, 3, 1).view(B, L, C)
        
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class TimeAlignBlock(nn.Module):
    def __init__(self, t_in, t_out, dim):
        super().__init__()
        self.time_proj = nn.Linear(t_in, t_out)
        self.norm = nn.LayerNorm(dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x: [B, T_in, C, H, W]
        x = x.permute(0, 2, 3, 4, 1).contiguous() # [B, C, H, W, T_in]
        x = self.time_proj(x) 
        x = x.permute(0, 4, 1, 2, 3).contiguous() # [B, T_out, C, H, W]
        
        B, T, C, H, W = x.shape
        x = x.view(B, T, C, -1).permute(0, 1, 3, 2) # [B, T, N, C]
        x = self.norm(x)
        x = x.permute(0, 1, 3, 2).view(B, T, C, H, W)
        return x

# ==========================================
# 核心 Mamba 模块
# ==========================================

class STMambaBlock(nn.Module):
    """
    Spatio-Temporal Mamba Block with Multi-directional Scan (H + V)
    """
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, 
                 d_state=16, d_conv=4, expand=2, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm1 = RMSNorm(dim)
        mamba_cfg = dict(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.scan = Mamba(**mamba_cfg)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = RMSNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = LocalityEnhancedMLP(dim, mlp_hidden_dim, dim, act_layer=act_layer, drop=drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, x_in):
        # x_in: [B, T, C, H, W]
        B, T, C, H, W = x_in.shape
        L = T * H * W
        
        # 1. Prepare sequence: [B, L, C] (Row-Major)
        x = x_in.permute(0, 1, 3, 4, 2).contiguous().reshape(B, L, C)
        
        def scan_multidir(z_row, B, T, H, W, C):
            """
            Internal function for checkpointing. 
            Explicitly passing dimensions ensures safety.
            """
            L = T * H * W
            
            # --- 1. Horizontal Scan ---
            # Forward + Backward
            out_h = self.scan(z_row) + self.scan(z_row.flip([1])).flip([1])
            
            # --- 2. Vertical Scan ---
            # [B, L, C] -> [B, T, H, W, C] -> [B, T, W, H, C]
            z_col = z_row.view(B, T, H, W, C).permute(0, 1, 3, 2, 4).contiguous().reshape(B, L, C)
            
            out_v = self.scan(z_col) + self.scan(z_col.flip([1])).flip([1])
            
            # Restore view: [B, T, W, H, C] -> [B, T, H, W, C] -> [B, L, C]
            out_v = out_v.view(B, T, W, H, C).permute(0, 1, 3, 2, 4).contiguous().reshape(B, L, C)
            
            return out_h + out_v

        # 2. Execution with Checkpoint Support
        shortcut = x
        x_norm = self.norm1(x)
        
        if self.use_checkpoint and x.requires_grad:
            # Pass scalar args as they don't require grad, but required for shape reconstruction
            out = checkpoint(scan_multidir, x_norm, B, T, H, W, C, use_reentrant=False)
        else:
            out = scan_multidir(x_norm, B, T, H, W, C)
            
        x = shortcut + self.drop_path(out)
        
        # 3. MLP with Locality
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        
        # 4. Restore: [B, T, C, H, W]
        return x.reshape(B, T, H, W, C).permute(0, 1, 4, 2, 3).contiguous()