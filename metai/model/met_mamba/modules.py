# metai/model/met_mamba/modules.py

import torch
import torch.nn as nn
import torch.nn.functional as F
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
# 1. 基础卷积与上采样组件
# ==========================================

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, 
                 dilation=1, upsampling=False, act_norm=False):
        super().__init__()
        self.act_norm = act_norm
        
        # 传统的上采样方式 (PixelShuffle)，保留用于 ConvSC 的兼容性
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
    """
    Skip-Connection Convolution Block
    用于 Encoder 和 Decoder 的基础层
    """
    def __init__(self, C_in, C_out, kernel_size=3, downsampling=False, upsampling=False, act_norm=True):
        super().__init__()
        stride = 2 if downsampling else 1
        padding = (kernel_size - stride + 1) // 2
        self.conv = BasicConv2d(C_in, C_out, kernel_size, stride, padding, upsampling=upsampling, act_norm=act_norm)

    def forward(self, x):
        return self.conv(x)

class ResizeConv(nn.Module):
    """
    [新增] Resize-Convolution 上采样模块
    
    使用 "双线性插值 + 卷积" 替代 PixelShuffle 或转置卷积。
    优势：能有效消除棋盘格伪影 (Checkerboard Artifacts)，产生更平滑的气象场预测。
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, act_norm=True):
        super().__init__()
        # 1. 双线性插值上采样 (Scale=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        # 2. 卷积修正特征
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
    """
    局部增强 MLP
    在 FFN 中引入深度可分离卷积 (Depth-wise Conv) 来增强局部空间感知能力。
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        # DW Conv: 增强局部性
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
        # x: [B, L, C] 
        # L 可能是 H*W (单帧) 或 T*H*W (多帧扁平化)
        x = self.fc1(x)
        B, L, C = x.shape
        
        # 自动推断 T 维度
        T = L // (H * W)
        
        # Reshape 为 (B*T, C, H, W) 进行 2D 卷积
        x_spatial = x.view(B * T, H, W, C).permute(0, 3, 1, 2)
        x_spatial = self.dwconv(x_spatial)
        
        # 还原回序列形状 [B, L, C]
        x = x_spatial.permute(0, 2, 3, 1).view(B, L, C)
        
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class TimeAlignBlock(nn.Module):
    """
    时间对齐模块
    用于 Skip Connection 中，将 Encoder 的时间步 T_in 投影到 Decoder 的 T_out。
    """
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
# 2. 核心 Mamba 模块 (Spatio-Temporal)
# ==========================================

class STMambaBlock(nn.Module):
    """
    [改进核心] STMambaBlock (Spatio-Temporal Mamba Block)
    
    不同于分别处理空间和时间的传统做法，本模块将时间维度 T 展开到序列长度中 
    (SeqLen = T * H * W)，利用 Mamba 的线性复杂度优势，
    一次性对整个时空体 (Spatio-Temporal Volume) 进行因果扫描。
    
    特性：
    1. 连续性：隐状态 (Hidden State) 可以在帧与帧之间传递。
    2. 双向性：采用正向+反向扫描，增强历史与未来信息的交互。
    3. 局部性：配合 LocalityEnhancedMLP 保持对空间纹理的感知。
    """
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, 
                 d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        
        # Mamba 核心配置
        mamba_cfg = dict(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.scan = Mamba(**mamba_cfg)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        
        # MLP 部分
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
        L = T * H * W
        
        # 1. 展开为长序列 [B, L, C]
        # 此时序列的物理意义是：第1帧全部像素 -> 第2帧全部像素 -> ... -> 第T帧
        x = x_in.permute(0, 1, 3, 4, 2).contiguous().reshape(B, L, C)
        
        shortcut = x
        x = self.norm1(x)
        
        # 2. 时空连续 Mamba 扫描
        # Forward Scan: 捕捉过去 -> 未来的演变
        # Backward Scan: 利用未来信息修正 (对于 Seq2Seq 生成任务很有帮助)
        out = self.scan(x) + self.scan(x.flip([1])).flip([1])
        
        x = shortcut + self.drop_path(out)
        
        # 3. 局部增强 MLP
        # LocalityEnhancedMLP 内部会将长序列 Reshape 为 (B*T, C, H, W) 
        # 进行 DW-Conv，从而提取每帧内部的空间纹理特征
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        
        # 4. 还原维度 [B, T, C, H, W]
        return x.reshape(B, T, H, W, C).permute(0, 1, 4, 2, 3).contiguous()