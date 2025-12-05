# metai/model/met_mamba/modules.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from timm.layers.drop import DropPath
from timm.layers.weight_init import trunc_normal_
from mamba_ssm import Mamba

# ==============================================================================
# 1. 基础组件 (Basic Components)
# ==============================================================================

class RMSNorm(nn.Module):
    """
    RMSNorm (Root Mean Square Layer Normalization) 归一化层。
    相比 LayerNorm 去除了均值中心化，计算更高效，常用于大模型。
    """
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return output * self.weight

class BasicConv2d(nn.Module):
    """
    基础卷积单元: Conv2d -> GroupNorm -> SiLU
    支持可选的上采样 (PixelShuffle)。
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, upsampling=False, act_norm=False):
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
        if self.act_norm: y = self.act(self.norm(y))
        return y

class ConvSC(nn.Module):
    """
    带残差连接的卷积块 (Skip-Connection Conv)。
    用于 Encoder 和 Decoder 的层级构建，支持下采样和上采样。
    """
    def __init__(self, C_in, C_out, kernel_size=3, downsampling=False, upsampling=False, act_norm=True):
        super().__init__()
        stride = 2 if downsampling else 1
        padding = (kernel_size - stride + 1) // 2
        self.conv = BasicConv2d(C_in, C_out, kernel_size, stride, padding, upsampling=upsampling, act_norm=act_norm)

    def forward(self, x): return self.conv(x)

class ResizeConv(nn.Module):
    """
    基于插值的上采样卷积块 (Resize + Conv)。
    相比转置卷积，能减少棋盘格效应。
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

class TimeAlignBlock(nn.Module):
    """
    时间对齐投影块。
    用于 U-Net 结构中 Skip Connection 的时间维度适配 (T_in -> T_out)。
    """
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
        # Input: [B, T_in, C, H, W]
        x = x.permute(0, 2, 3, 4, 1).contiguous() # -> [B, C, H, W, T_in]
        x = self.time_proj(x) # -> [B, C, H, W, T_out]
        x = x.permute(0, 4, 1, 2, 3).contiguous() # -> [B, T_out, C, H, W]
        
        # LayerNorm over Channel dimension
        B, T, C, H, W = x.shape
        x = x.view(B, T, C, -1).permute(0, 1, 3, 2)
        x = self.norm(x)
        x = x.permute(0, 1, 3, 2).view(B, T, C, H, W)
        return x

# ==============================================================================
# 2. CVAE 组件 (Probabilistic Generation)
# ==============================================================================

class DistributionEncoder(nn.Module):
    """
    分布参数编码头。
    将提取的高维特征映射到高斯分布的参数 (均值 mu, 对数方差 logvar)。
    """
    def __init__(self, in_dim, latent_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU()
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_var = nn.Linear(256, latent_dim)

    def forward(self, x):
        # x: [B, in_dim]
        h = self.net(x)
        mu = self.fc_mu(h)
        logvar = self.fc_var(h)
        return mu, logvar

class PosteriorNet(nn.Module):
    """
    后验/先验编码网络 (Posterior/Prior Network)。
    使用 3D 卷积从时空序列中提取全局上下文特征，用于估计隐变量 z 的分布。
    
    - 训练时作为 Posterior Q(z|X,Y): 输入包含未来的全序列。
    - 推理时作为 Prior P(z|X): 输入仅包含历史序列。
    """
    def __init__(self, in_channels, latent_dim=32):
        super().__init__()
        # 3D CNN 用于提取时空特征 (Time-Space Feature Extraction)
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1),
            nn.GroupNorm(16, 128),
            nn.SiLU(),
            nn.AdaptiveAvgPool3d(1), # 全局时空池化，提取唯一的 Context Vector
            nn.Flatten()
        )
        self.dist_head = DistributionEncoder(128, latent_dim)

    def forward(self, x_seq):
        # x_seq: [B, C, T, H, W]
        feat = self.encoder(x_seq)
        return self.dist_head(feat)

# ==============================================================================
# 3. 核心算子: 空间 Mamba 与 软门控 (Spatial Mamba & Soft-Gating)
# ==============================================================================

class SpatialMamba2D(nn.Module):
    """
    双向 2D 空间 Mamba 模块。
    
    机制：
    通过将 2D 图像展平为序列，利用 Mamba 的线性复杂度 O(N) 进行全局建模。
    采用双向扫描 (正向 + 反向) 消除因展平顺序引入的因果偏差，替代传统的 Self-Attention。
    """
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        
        # 定义两个方向的 SSM (State Space Model)
        self.mamba_forward = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_backward = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.proj_out = nn.Linear(dim * 2, dim)

    def forward(self, x):
        # x: [B, T, H, W, C]
        B, T, H, W, C = x.shape
        x_flat = x.view(B * T, H * W, C)
        x_norm = self.norm(x_flat)

        # 1. 正向扫描 (Row-Major)
        out_fwd = self.mamba_forward(x_norm)
        
        # 2. 反向扫描 (Row-Major Reversed)
        # 通过翻转序列 -> Mamba -> 再翻转回来，捕捉反向的空间依赖
        out_bwd = self.mamba_backward(x_norm.flip(dims=[1])).flip(dims=[1])
        
        # 3. 特征融合
        out = torch.cat([out_fwd, out_bwd], dim=-1)
        out = self.proj_out(out)
        
        return out.view(B, T, H, W, C)

class SoftSparseGate(nn.Module):
    """
    可微分软门控 (Soft-Gating) 模块。
    
    机制：
    生成一个空间重要性图 (Importance Map, 0~1)，对特征进行加权。
    优势：
    相比硬截断 (Top-K)，软门控保证了全图梯度流的完整性，防止稀疏路由在训练初期坍塌。
    """
    def __init__(self, dim):
        super().__init__()
        self.gate_conv = nn.Sequential(
            nn.Conv2d(dim, dim // 4, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(dim // 4, 1, kernel_size=1),
            nn.Sigmoid() 
        )

    def forward(self, x):
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        x_in = x.view(B * T, C, H, W)
        
        # 生成重要性权重
        importance_map = self.gate_conv(x_in) # [B*T, 1, H, W]
        
        return importance_map.view(B, T, 1, H, W)

# ==============================================================================
# 4. 时空状态空间模块 (Core Block)
# ==============================================================================

class STMambaBlock(nn.Module):
    """
    MeteoMamba 核心模块 (Spatio-Temporal Mamba Block)。
    
    处理流程:
    1. Temporal Mamba: 处理每个像素点的时序演变 (1D SSM)。
    2. Soft Gating: 抑制背景噪声，聚焦强回波区域 (Sparse Mechanism)。
    3. Spatial Mamba: 处理全图的空间长程依赖 (2D Bi-SSM)。
    """
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, drop_path=0., use_checkpoint=False, **kwargs):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        
        # 1. 时间演变分支
        self.norm_t = nn.LayerNorm(dim)
        self.mamba_t = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        
        # 2. 空间建模分支
        self.space_block = SpatialMamba2D(dim, d_state=d_state, d_conv=d_conv, expand=expand)
        
        # 3. 软稀疏门控
        self.sparse_gate = SoftSparseGate(dim)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        
        # === 步骤 A: 时间演变 (Temporal Evolution) ===
        # 维度变换: [B, T, C, H, W] -> [B*HW, T, C]
        x_t_in = x.permute(0, 3, 4, 1, 2).reshape(B * H * W, T, C)
        x_t_in = self.norm_t(x_t_in)
        
        if self.use_checkpoint and x_t_in.requires_grad:
            x_t_out = checkpoint(self.mamba_t, x_t_in, use_reentrant=False)
        else:
            x_t_out = self.mamba_t(x_t_in)
            
        x_t_out = x_t_out.view(B, H, W, T, C).permute(0, 3, 4, 1, 2)
        x = x + self.drop_path(x_t_out)
        
        # === 步骤 B: 软稀疏加权 (Soft Sparsity) ===
        imp_map = self.sparse_gate(x) # [B, T, 1, H, W]
        # 残差式加权：x * (0.5 + 0.5 * map)，保留 0.5 的底数防止特征完全被抑制
        x_weighted = x * (0.5 + 0.5 * imp_map)
        
        # === 步骤 C: 空间混合 (Spatial Mixing) ===
        # 维度变换: [B, T, C, H, W] -> [B, T, H, W, C]
        x_s_in = x_weighted.permute(0, 1, 3, 4, 2)
        x_s_out = self.space_block(x_s_in)
        x_s_out = x_s_out.permute(0, 1, 4, 2, 3) # -> [B, T, C, H, W]
        
        x = x + self.drop_path(x_s_out)
        
        return x

# ==============================================================================
# 5. 物理平流组件 (Physical Advection)
# ==============================================================================

class FlowPredictorGRU(nn.Module):
    """
    光流预测 GRU 网络。
    用于递归地预测未来的流场 (Flow Field) 及其残差。
    """
    def __init__(self, in_channels, hidden_dim):
        super().__init__()
        self.conv_z = nn.Conv2d(in_channels + hidden_dim, hidden_dim, 3, 1, 1)
        self.conv_r = nn.Conv2d(in_channels + hidden_dim, hidden_dim, 3, 1, 1)
        self.conv_h = nn.Conv2d(in_channels + hidden_dim, hidden_dim, 3, 1, 1)
        self.flow_head = nn.Conv2d(hidden_dim, 2, 3, 1, 1)
        self.residual_head = nn.Conv2d(hidden_dim, in_channels, 3, 1, 1)

        # 初始化为 0，使初始流场静止，便于训练稳定
        nn.init.constant_(self.flow_head.weight, 0)
        nn.init.constant_(self.flow_head.bias, 0)

    def forward(self, x, h):
        combined = torch.cat([x, h], dim=1)
        z = torch.sigmoid(self.conv_z(combined))
        r = torch.sigmoid(self.conv_r(combined))
        combined_new = torch.cat([x, r * h], dim=1)
        h_tilde = torch.tanh(self.conv_h(combined_new))
        h_next = (1 - z) * h + z * h_tilde
        
        delta_flow = self.flow_head(h_next)
        residual = self.residual_head(h_next)
        return delta_flow, residual, h_next

class AdvectiveProjection(nn.Module):
    """
    显式物理平流层 (Advective Projection Layer)。
    
    机制：
    基于光流方程 (Optical Flow)，利用历史最后一帧作为初始状态，
    递归地推演未来的平流运动，提供物理上的“保底”预测，防止第二帧预测迅速衰减。
    """
    def __init__(self, dim, t_in, t_out):
        super().__init__()
        self.t_in = t_in
        self.t_out = t_out
        self.dim = dim
        self.flow_predictor = FlowPredictorGRU(in_channels=dim, hidden_dim=dim)
        self.init_h = nn.Conv2d(dim, dim, 1)

    def forward(self, z_seq):
        # z_seq: [B, T_in, C, H, W]
        B, T_in, C, H, W = z_seq.shape
        # 取历史序列的最后一帧作为平流起点
        curr_z = z_seq[:, -1]
        h = self.init_h(curr_z)
        
        # 建立归一化采样网格 (Normalized Grid)
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H, device=z_seq.device),
            torch.linspace(-1, 1, W, device=z_seq.device),
            indexing='ij'
        )
        base_grid = torch.stack([xx, yy], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
        
        preds = []
        flow_maps = [] 
        
        for i in range(self.t_out):
            # 预测流场增量和特征残差
            flow_delta, residual, h = self.flow_predictor(curr_z, h)
            flow_maps.append(flow_delta) 
            
            flow_field = flow_delta.permute(0, 2, 3, 1)
            grid = base_grid + flow_field
            
            # 物理平流操作 (Grid Sample / Warp)
            z_advected = F.grid_sample(
                curr_z, grid, mode='bilinear', padding_mode='reflection', align_corners=False
            )
            
            # 预测状态更新：平流结果 + 局部生消残差
            next_z = z_advected + residual
            curr_z = next_z
            preds.append(curr_z)
            
        return torch.stack(preds, dim=1), torch.stack(flow_maps, dim=1)