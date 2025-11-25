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

class TimeAlignBlock(nn.Module):
    """时序对齐模块：将 T_in 帧的特征映射到 T_out 帧"""
    def __init__(self, t_in, t_out, dim):
        super().__init__()
        self.time_proj = nn.Linear(t_in, t_out)
        self.norm = nn.LayerNorm(dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.01)
            # 关键初始化：默认未来是现在的延续
            with torch.no_grad():
                if m.weight.shape[1] > 0:
                    m.weight[:, -1].fill_(1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # x: [B, T_in, C, H, W] -> [B, C, H, W, T_in]
        x = x.permute(0, 2, 3, 4, 1).contiguous() 
        x = self.time_proj(x) # -> [B, C, H, W, T_out]
        x = x.permute(0, 4, 1, 2, 3).contiguous() # -> [B, T_out, C, H, W]
        
        # Apply Norm
        B, T, C, H, W = x.shape
        x = x.view(B, T, C, -1).permute(0, 1, 3, 2) 
        x = self.norm(x)
        x = x.permute(0, 1, 3, 2).view(B, T, C, H, W)
        return x

class STMambaBlock(nn.Module):
    """时空曼巴块：结合 SS2D 和 Temporal Scan"""
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        
        mamba_cfg = dict(d_model=dim, d_state=16, d_conv=4, expand=2)
        self.scan_spatial = Mamba(**mamba_cfg)
        self.scan_temporal = Mamba(**mamba_cfg)
        
        self.fusion = nn.Linear(dim * 2, dim)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )
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
        shortcut = x
        x = self.norm1(x)
        
        # Branch 1: Spatial Scan (Batch = B*T)
        x_s = x.reshape(B*T, N, C)
        out_s = self.scan_spatial(x_s).reshape(B, T, N, C)
        
        # Branch 2: Temporal Scan (Batch = B*N)
        x_t = x.permute(0, 2, 1, 3).reshape(B*N, T, C)
        out_t = self.scan_temporal(x_t).reshape(B, N, T, C).permute(0, 2, 1, 3)
        
        # Fusion
        fused = torch.cat([out_s, out_t], dim=-1)
        x = self.fusion(fused)
        
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        # Restore: [B, T, C, H, W]
        return x.reshape(B, T, H, W, C).permute(0, 1, 4, 2, 3)