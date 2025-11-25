import torch
import torch.nn as nn
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
            raise ImportError("Please install 'mamba_ssm' (pip install mamba-ssm)")

# ==========================================
# 1. 基础组件
# ==========================================
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
            nn.init.constant_(m.bias, 0)

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

# ==========================================
# 2. 高级组件 (SOTA Optimizations)
# ==========================================
class TokenSpaceMLP(nn.Module):
    """
    [Optimization] Token Space MLP
    直接在 Token 维度 [B, L, C] 操作，避免频繁的 permute/reshape。
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class AdaptiveFusionGate(nn.Module):
    """
    [Optimization] Adaptive Fusion Gate
    基于上下文内容，动态学习空间特征与时间特征的融合权重。
    """
    def __init__(self, dim, reduction=4):
        super().__init__()
        gate_dim = max(dim // reduction, 8)
        self.gate_net = nn.Sequential(
            nn.Linear(dim, gate_dim),
            nn.ReLU(inplace=True),
            nn.Linear(gate_dim, dim),
            nn.Sigmoid()
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        
    def forward(self, x_spatial, x_temporal):
        # x: [B, L, C]
        # Global Context: Average over tokens
        combined = (x_spatial + x_temporal) / 2.0
        context = combined.mean(dim=1) # [B, C]
        
        gate = self.gate_net(context).unsqueeze(1) # [B, 1, C]
        return gate * x_spatial + (1 - gate) * x_temporal

# ==========================================
# 3. 核心模块 (Core Blocks)
# ==========================================
class TimeAlignBlock(nn.Module):
    """时序对齐模块：T_in -> T_out 投影"""
    def __init__(self, t_in, t_out, dim):
        super().__init__()
        self.time_proj = nn.Linear(t_in, t_out)
        self.norm = nn.LayerNorm(dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.01)
            with torch.no_grad():
                if m.weight.shape[1] > 0:
                    # Init: Future is copy of last frame
                    m.weight[:, -1].fill_(1.0)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # x: [B, T_in, C, H, W]
        x = x.permute(0, 2, 3, 4, 1).contiguous() 
        x = self.time_proj(x) 
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        
        # Norm
        B, T, C, H, W = x.shape
        x = x.view(B, T, C, -1).permute(0, 1, 3, 2) 
        x = self.norm(x)
        x = x.permute(0, 1, 3, 2).view(B, T, C, H, W)
        return x

class STMambaBlock(nn.Module):
    """
    Spatiotemporal Mamba Block
    Features:
    1. Bidirectional Scanning (Fwd + Bwd) for both Spatial and Temporal branches.
    2. Adaptive Fusion Gate.
    3. TokenSpaceMLP.
    """
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        
        mamba_cfg = dict(d_model=dim, d_state=16, d_conv=4, expand=2)
        
        # Branch 1: Spatial (SS2D-like)
        self.scan_spatial = Mamba(**mamba_cfg)
        
        # Branch 2: Temporal
        self.scan_temporal = Mamba(**mamba_cfg)
        
        # SOTA: Adaptive Gate
        self.fusion_gate = AdaptiveFusionGate(dim)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        
        # SOTA: Token Space MLP
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = TokenSpaceMLP(dim, mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x_in):
        # x_in: [B, T, C, H, W]
        B, T, C, H, W = x_in.shape
        N = H * W
        
        # Flatten to Token Space: [B, T, N, C]
        x = x_in.permute(0, 1, 3, 4, 2).reshape(B, T, N, C)
        shortcut = x.reshape(B, T*N, C)
        x = self.norm1(x) # [B, T, N, C]
        
        # --- Branch 1: Spatial Scanning ---
        # Reshape: [B*T, N, C]
        x_s = x.reshape(B*T, N, C)
        # Bidirectional: Fwd + Bwd
        out_s_fwd = self.scan_spatial(x_s)
        out_s_bwd = self.scan_spatial(x_s.flip([1])).flip([1])
        out_s = (out_s_fwd + out_s_bwd).reshape(B, T*N, C)
        
        # --- Branch 2: Temporal Scanning ---
        # Reshape: [B*N, T, C]
        x_t = x.permute(0, 2, 1, 3).reshape(B*N, T, C)
        # Bidirectional: Fwd + Bwd
        out_t_fwd = self.scan_temporal(x_t)
        out_t_bwd = self.scan_temporal(x_t.flip([1])).flip([1])
        out_t = (out_t_fwd + out_t_bwd).reshape(B, N, T, C).permute(0, 2, 1, 3).reshape(B, T*N, C)
        
        # --- Adaptive Fusion ---
        x = self.fusion_gate(out_s, out_t)
        
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        # Restore: [B, T, C, H, W]
        return x.reshape(B, T, H, W, C).permute(0, 1, 4, 2, 3)