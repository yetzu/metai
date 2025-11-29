# metai/model/met_mamba/modules.py

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
            raise ImportError("Please install 'mamba_ssm'")

# ==========================================
# 1. 基础组件 (保持不变)
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
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, x, H, W):
        x = self.fc1(x)
        B, L, C = x.shape
        T = L // (H * W)
        x_spatial = x.view(B * T, H, W, C).permute(0, 3, 1, 2)
        x_spatial = self.dwconv(x_spatial)
        x = x_spatial.permute(0, 2, 3, 1).view(B, L, C)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class AdaptiveFusionGate(nn.Module):
    """
    [保留组件] 自适应融合门
    现在用于融合 SS2D 的水平流 (H) 和垂直流 (V)
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
        
    def forward(self, branch1, branch2):
        # Global Context: Average over tokens
        combined = (branch1 + branch2) / 2.0
        context = combined.mean(dim=1) # [B, C]
        gate = self.gate_net(context).unsqueeze(1) # [B, 1, C]
        return gate * branch1 + (1 - gate) * branch2

class TimeAlignBlock(nn.Module):
    def __init__(self, t_in, t_out, dim):
        super().__init__()
        self.time_proj = nn.Linear(t_in, t_out)
        self.norm = nn.LayerNorm(dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.01)
            with torch.no_grad():
                if m.weight.shape[1] > 0: m.weight[:, -1].fill_(1.0)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = x.permute(0, 2, 3, 4, 1).contiguous() 
        x = self.time_proj(x) 
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        B, T, C, H, W = x.shape
        x = x.view(B, T, C, -1).permute(0, 1, 3, 2) 
        x = self.norm(x)
        x = x.permute(0, 1, 3, 2).view(B, T, C, H, W)
        return x

# ==========================================
# 2. 优化后的核心模块 (Spatial & Temporal)
# ==========================================

class SpatialMambaBlock(nn.Module):
    """
    [SS2D] 空间 Mamba 块：水平+垂直双向扫描
    使用 AdaptiveFusionGate 融合 H 和 V 分支
    """
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, 
                 d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        mamba_cfg = dict(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        
        self.scan_h = Mamba(**mamba_cfg)
        self.scan_v = Mamba(**mamba_cfg)
        
        # [保留] 使用 FusionGate 融合 H 和 V 的结果
        self.fusion_gate = AdaptiveFusionGate(dim)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = LocalityEnhancedMLP(dim, mlp_hidden_dim, dim, act_layer=act_layer, drop=drop)
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
        
        x = x_in.permute(0, 1, 3, 4, 2).contiguous().reshape(B, T, N, C)
        shortcut = x.reshape(B, T*N, C)
        x = self.norm1(x)
        
        # SS2D Logic
        x_s = x.reshape(B*T, H, W, C)
        
        # 1. Horizontal
        x_h = x_s.reshape(B*T, N, C)
        out_h = self.scan_h(x_h) + self.scan_h(x_h.flip([1])).flip([1])
        out_h = out_h.reshape(B, T*N, C)
        
        # 2. Vertical
        x_v = x_s.permute(0, 2, 1, 3).contiguous().reshape(B*T, N, C)
        out_v = self.scan_v(x_v) + self.scan_v(x_v.flip([1])).flip([1])
        # Restore vertical order
        out_v = out_v.reshape(B*T, W, H, C).permute(0, 2, 1, 3).contiguous().reshape(B, T*N, C)
        
        # 3. Adaptive Fusion (H + V)
        x = self.fusion_gate(out_h, out_v)
        
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        
        return x.reshape(B, T, H, W, C).permute(0, 1, 4, 2, 3).contiguous()

class TemporalMambaBlock(nn.Module):
    """
    时间 Mamba 块：时间轴双向扫描
    """
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, 
                 d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        mamba_cfg = dict(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        
        self.scan_t = Mamba(**mamba_cfg)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = LocalityEnhancedMLP(dim, mlp_hidden_dim, dim, act_layer=act_layer, drop=drop)
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
        
        x = x_in.permute(0, 1, 3, 4, 2).reshape(B, T, N, C)
        shortcut = x.reshape(B, T*N, C)
        x = self.norm1(x)
        
        # Temporal Scan
        x_t = x.permute(0, 2, 1, 3).contiguous().reshape(B*N, T, C)
        out_t = self.scan_t(x_t) + self.scan_t(x_t.flip([1])).flip([1])
        out_t = out_t.reshape(B, N, T, C).permute(0, 2, 1, 3).contiguous().reshape(B, T*N, C)
        
        x = shortcut + self.drop_path(out_t)
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        
        return x.reshape(B, T, H, W, C).permute(0, 1, 4, 2, 3).contiguous()