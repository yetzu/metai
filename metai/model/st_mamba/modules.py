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
# 2. SOTA 组件 (LocalityEnhancedMLP & AdaptiveGate)
# ==========================================
class LocalityEnhancedMLP(nn.Module):
    """
    [Optimization] Locality Enhanced MLP (原 TokenSpaceMLP 升级版)
    
    改进点：
    在 Token Space 的 Linear 层之间引入 3x3 Depth-wise Conv。
    Mamba 擅长全局依赖，DWConv 补充局部纹理细节，这对短临降水预测非常关键。
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        
        # [New] Depth-wise Convolution for Locality
        # groups=hidden_features makes it depth-wise (channel-independent)
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
        """
        Args:
            x: [B, L, C] where L = T*H*W
            H, W: Spatial dimensions for reshaping
        """
        # 1. Projection
        x = self.fc1(x) # [B, L, hidden]
        
        # 2. Locality Enhancement (DWConv)
        B, L, C = x.shape
        T = L // (H * W)
        
        # Reshape: [B, T*H*W, C] -> [B*T, C, H, W]
        # 将 Batch 和 Time 维度合并处理，对每帧进行空间卷积
        x_spatial = x.view(B * T, H, W, C).permute(0, 3, 1, 2)
        
        x_spatial = self.dwconv(x_spatial)
        
        # Flatten back: [B*T, C, H, W] -> [B, L, C]
        x = x_spatial.permute(0, 2, 3, 1).view(B, L, C)
        
        # 3. Activation & Output
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class AdaptiveFusionGate(nn.Module):
    """
    Adaptive Fusion Gate: 动态学习空间与时间分支的融合权重
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
        # Global Context: Average over tokens
        combined = (x_spatial + x_temporal) / 2.0
        context = combined.mean(dim=1) # [B, C]
        gate = self.gate_net(context).unsqueeze(1) # [B, 1, C]
        return gate * x_spatial + (1 - gate) * x_temporal

# ==========================================
# 3. 核心模块 (TimeAlign & STMamba)
# ==========================================
class TimeAlignBlock(nn.Module):
    """时序对齐模块：将 T_in 帧特征线性投影到 T_out 帧"""
    def __init__(self, t_in, t_out, dim):
        super().__init__()
        self.time_proj = nn.Linear(t_in, t_out)
        self.norm = nn.LayerNorm(dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.01)
            # Init: Future is mostly copy of last frame (Identity-like)
            with torch.no_grad():
                if m.weight.shape[1] > 0:
                    m.weight[:, -1].fill_(1.0)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # x: [B, T_in, C, H, W] -> [B, C, H, W, T_in]
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
    [SOTA] Spatiotemporal Mamba Block (Bidirectional)
    结合双向空间扫描 (Spatial) 和 双向时间扫描 (Temporal) + 自适应融合。
    优化：Locality Enhanced MLP 和 扩展的 State Dimension。
    """
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        
        # [Optimization] State Expansion
        # 将 d_state 从 16 提升至 32，增加 SSM 记忆容量，捕捉更长时序依赖
        mamba_cfg = dict(d_model=dim, d_state=32, d_conv=4, expand=2)
        
        # Branch 1: Spatial
        self.scan_spatial = Mamba(**mamba_cfg)
        
        # Branch 2: Temporal
        self.scan_temporal = Mamba(**mamba_cfg)
        
        # Adaptive Gate
        self.fusion_gate = AdaptiveFusionGate(dim)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        
        # [Optimization] 使用 LocalityEnhancedMLP 替代纯 Linear MLP
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
        
        # Flatten to Token Space: [B, T, N, C]
        x = x_in.permute(0, 1, 3, 4, 2).reshape(B, T, N, C)
        shortcut = x.reshape(B, T*N, C)
        x = self.norm1(x) # [B, T, N, C]
        
        # --- Branch 1: Spatial Scanning (Bidirectional) ---
        # Reshape: [B*T, N, C]
        x_s = x.reshape(B*T, N, C)
        out_s_fwd = self.scan_spatial(x_s)
        out_s_bwd = self.scan_spatial(x_s.flip([1])).flip([1])
        out_s = (out_s_fwd + out_s_bwd).reshape(B, T*N, C)
        
        # --- Branch 2: Temporal Scanning (Bidirectional) ---
        # Reshape: [B*N, T, C]
        x_t = x.permute(0, 2, 1, 3).reshape(B*N, T, C)
        out_t_fwd = self.scan_temporal(x_t)
        out_t_bwd = self.scan_temporal(x_t.flip([1])).flip([1])
        out_t = (out_t_fwd + out_t_bwd).reshape(B, N, T, C).permute(0, 2, 1, 3).reshape(B, T*N, C)
        
        # --- Adaptive Fusion ---
        x = self.fusion_gate(out_s, out_t)
        
        x = shortcut + self.drop_path(x)
        
        # [Optimization] Pass spatial dimensions H, W to MLP for DWConv
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        
        # Restore: [B, T, C, H, W]
        return x.reshape(B, T, H, W, C).permute(0, 1, 4, 2, 3)