# metai/model/met_mamba/modules.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from timm.layers.drop import DropPath
from timm.layers.weight_init import trunc_normal_

from mamba_ssm import Mamba

# ==========================================
# 基础组件 (Basic Components)
# ==========================================

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm)
    相比 LayerNorm，它去除了均值中心化操作，计算更高效且在 Transformer/Mamba 类架构中表现更稳定。
    """
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        # 计算均方根并进行归一化，最后应用可学习的缩放参数 weight
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return output * self.weight

class BasicConv2d(nn.Module):
    """
    基础 2D 卷积块
    集成：Conv2d -> GroupNorm -> SiLU
    支持通过 PixelShuffle 进行上采样。
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, 
                 dilation=1, upsampling=False, act_norm=False):
        super().__init__()
        self.act_norm = act_norm
        
        # 如果启用上采样，先将通道数扩充4倍，再通过 PixelShuffle(2) 变为 2倍宽高
        if upsampling:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels*4, kernel_size, stride, padding, dilation),
                nn.PixelShuffle(2)
            )
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation)
        
        # 使用 GroupNorm(2 groups) 进行归一化
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
    带步长控制的卷积模块 (Convolution with Stride Control)
    用于 Encoder/Decoder 中控制特征图分辨率的缩放。
    """
    def __init__(self, C_in, C_out, kernel_size=3, downsampling=False, upsampling=False, act_norm=True):
        super().__init__()
        # 下采样时步长为2，否则为1
        stride = 2 if downsampling else 1
        # 自动计算 padding 以保持或正确缩放尺寸
        padding = (kernel_size - stride + 1) // 2
        self.conv = BasicConv2d(C_in, C_out, kernel_size, stride, padding, upsampling=upsampling, act_norm=act_norm)

    def forward(self, x):
        return self.conv(x)

class ResizeConv(nn.Module):
    """
    抗伪影上采样模块 (Resize-Convolution)
    使用 "双线性插值 + 卷积" 替代 PixelShuffle 或转置卷积。
    目的：消除生成图像中的棋盘格伪影 (checkerboard artifacts)，这对气象云图生成尤为重要。
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, act_norm=True):
        super().__init__()
        # 1. 双线性插值放大2倍
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        padding = kernel_size // 2
        # 2. 卷积层调整特征
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
    局部增强型多层感知机 (Locality Enhanced MLP)
    在标准 MLP 中间插入深度可分离卷积 (Depth-wise Conv)，
    用于在 Channel 混合的同时捕获局部空间上下文信息。
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        # 1. 升维
        self.fc1 = nn.Linear(in_features, hidden_features)
        # 2. 深度卷积 (DWConv): 捕获局部特征，groups=hidden_features 表示逐通道卷积
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)
        self.act = act_layer()
        # 3. 降维
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, x, H, W):
        # 输入 x: [B, L, C]  (L = T*H*W)
        x = self.fc1(x)
        B, L, C = x.shape
        T = L // (H * W)
        
        # 变换维度以进行 2D 卷积: [B, L, C] -> [B*T, C, H, W]
        # 注意：这里将 Batch 和 Time 合并，对每一帧独立进行空间卷积
        x_spatial = x.view(B * T, H, W, C).permute(0, 3, 1, 2)
        x_spatial = self.dwconv(x_spatial)
        
        # 还原维度: [B*T, C, H, W] -> [B, L, C]
        x = x_spatial.permute(0, 2, 3, 1).view(B, L, C)
        
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class TimeAlignBlock(nn.Module):
    """
    时间对齐模块 (Time Alignment Block)
    用于在时间维度上投影，将输入序列长度映射到输出序列长度。
    常用于 Skip Connection 或 Output Layer 中对齐 T_in 和 T_out。
    """
    def __init__(self, t_in, t_out, dim):
        super().__init__()
        # 在时间维度上进行线性投影
        self.time_proj = nn.Linear(t_in, t_out)
        self.norm = nn.LayerNorm(dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 输入 x: [B, T_in, C, H, W]
        # 调整维度将 T_in 放到最后: [B, C, H, W, T_in]
        x = x.permute(0, 2, 3, 4, 1).contiguous() 
        # 执行投影: T_in -> T_out
        x = self.time_proj(x) 
        # 恢复维度: [B, T_out, C, H, W]
        x = x.permute(0, 4, 1, 2, 3).contiguous() 
        
        # LayerNorm (针对 Channel 维度)
        B, T, C, H, W = x.shape
        x = x.view(B, T, C, -1).permute(0, 1, 3, 2) # [B, T, N, C]
        x = self.norm(x)
        x = x.permute(0, 1, 3, 2).view(B, T, C, H, W)
        return x

# ==========================================
# 核心 Mamba 模块 (Core Mamba Modules)
# ==========================================

class STMambaBlock(nn.Module):
    """
    时空 Mamba 模块 (Spatio-Temporal Mamba Block)
    核心特性：通过多向扫描 (水平 + 垂直) 来捕捉 2D 空间依赖。
    """
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, 
                 d_state=16, d_conv=4, expand=2, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm1 = RMSNorm(dim)
        
        # Mamba 核心参数配置
        mamba_cfg = dict(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.scan = Mamba(**mamba_cfg)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = RMSNorm(dim)
        
        # 使用局部增强 MLP
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = LocalityEnhancedMLP(dim, mlp_hidden_dim, dim, act_layer=act_layer, drop=drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, x_in):
        # 输入 x_in: [B, T, C, H, W]
        B, T, C, H, W = x_in.shape
        L = T * H * W
        
        # 1. 准备序列: 将 3D (T, H, W) 展平为 1D 序列 [B, L, C] (行优先 Row-Major)
        x = x_in.permute(0, 1, 3, 4, 2).contiguous().reshape(B, L, C)
        
        def scan_multidir(z_row, B, T, H, W, C):
            """
            多向扫描内部函数，支持梯度检查点 (Checkpointing)。
            显式传递维度参数以确保重构形状时的安全性。
            """
            L = T * H * W
            
            # --- 1. 水平扫描 (Horizontal Scan) ---
            # 原始序列即为行优先，直接扫描为前向；翻转后扫描为后向。
            # 结果 = Forward(Row) + Backward(Row)
            out_h = self.scan(z_row) + self.scan(z_row.flip([1])).flip([1])
            
            # --- 2. 垂直扫描 (Vertical Scan) ---
            # 变换视图以模拟列优先遍历 (Column-Major)
            # [B, L, C] -> [B, T, H, W, C] -> [B, T, W, H, C]
            # 交换 H 和 W 维度，使得在展平后，相邻元素在原始图像中是垂直相邻的
            z_col = z_row.view(B, T, H, W, C).permute(0, 1, 3, 2, 4).contiguous().reshape(B, L, C)
            
            # 垂直方向的前向 + 后向扫描
            out_v = self.scan(z_col) + self.scan(z_col.flip([1])).flip([1])
            
            # 还原视图: [B, T, W, H, C] -> [B, T, H, W, C] -> [B, L, C]
            # 必须将维度 permute 回去，以便与 out_h 形状对齐进行相加
            out_v = out_v.view(B, T, W, H, C).permute(0, 1, 3, 2, 4).contiguous().reshape(B, L, C)
            
            # 融合水平和垂直扫描的信息
            return out_h + out_v

        # 2. 执行扫描 (带 Checkpoint 支持)
        shortcut = x
        x_norm = self.norm1(x)
        
        if self.use_checkpoint and x.requires_grad:
            # 传递标量参数 B, T, H, W, C 不需要梯度，但重构 View 需要它们
            out = checkpoint(scan_multidir, x_norm, B, T, H, W, C, use_reentrant=False)
        else:
            out = scan_multidir(x_norm, B, T, H, W, C)
            
        x = shortcut + self.drop_path(out)
        
        # 3. 局部增强 MLP
        # 传入 H, W 以便 MLP 内部还原空间结构进行卷积
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        
        # 4. 还原为原始 5D 形状: [B, T, C, H, W]
        return x.reshape(B, T, H, W, C).permute(0, 1, 4, 2, 3).contiguous()