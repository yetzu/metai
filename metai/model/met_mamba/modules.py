# metai/model/met_mamba/modules.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from timm.layers.drop import DropPath
from timm.layers.weight_init import trunc_normal_
from mamba_ssm import Mamba

# ==========================================
# 基础组件 (Basic Components) - 保持不变
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
    def __init__(self, C_in, C_out, kernel_size=3, downsampling=False, upsampling=False, act_norm=True):
        super().__init__()
        stride = 2 if downsampling else 1
        padding = (kernel_size - stride + 1) // 2
        self.conv = BasicConv2d(C_in, C_out, kernel_size, stride, padding, upsampling=upsampling, act_norm=act_norm)

    def forward(self, x): return self.conv(x)

class ResizeConv(nn.Module):
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
        x = x.permute(0, 2, 3, 4, 1).contiguous() # -> [B, C, H, W, T_in]
        x = self.time_proj(x) # -> [B, C, H, W, T_out]
        x = x.permute(0, 4, 1, 2, 3).contiguous() # -> [B, T_out, C, H, W]
        B, T, C, H, W = x.shape
        x = x.view(B, T, C, -1).permute(0, 1, 3, 2)
        x = self.norm(x)
        x = x.permute(0, 1, 3, 2).view(B, T, C, H, W)
        return x

# ==========================================
# 稀疏计算组件 (Structured Sparse Components) - [REFACTORED]
# ==========================================

class LearnableSparseHandler(nn.Module):
    """
    [改进版 v2.0] 鲁棒可微稀疏处理器
    特性：
    1. Mean+Max 双路特征融合：既保留背景动力学(Mean)，又捕捉强对流核心(Max)。
    2. Additive STE：使用加法直通估计器，解决除法导致的数值不稳定和梯度爆炸问题。
    """
    def __init__(self, dim, sparse_ratio=0.5):
        super().__init__()
        self.sparse_ratio = sparse_ratio
        
        # [NEW] 路由特征预处理：融合 Mean 和 Max 特征
        self.router_preprocess = nn.Sequential(
            nn.Conv2d(dim * 2, dim // 2, kernel_size=1), 
            nn.GroupNorm(4, dim // 2),
            nn.LeakyReLU(inplace=True)
        )
        
        self.router = nn.Sequential(
            nn.Conv2d(dim // 2, dim // 4, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(dim // 4, 1, kernel_size=1)
        )
        
        self.register_buffer('noise_scale', torch.tensor(0.01))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)

    def sparsify(self, x):
        B, T, C, H, W = x.shape
        N_spatial = H * W
        
        # [STEP 1] 提取全局感知特征 (Mean + Max)
        x_mean = x.mean(dim=1)  # [B, C, H, W]
        x_max, _ = x.max(dim=1) # [B, C, H, W]
        x_feat = torch.cat([x_mean, x_max], dim=1) # [B, 2C, H, W]
        
        # [STEP 2] 生成路由分数
        x_feat = self.router_preprocess(x_feat)
        scores_map = torch.sigmoid(self.router(x_feat)) # [B, 1, H, W]
        scores_flat = scores_map.view(B, N_spatial)
        
        # 训练时注入噪声以增强鲁棒性 (可选)
        if self.training:
            noise = torch.randn_like(scores_flat) * self.noise_scale
            scores_flat_noisy = scores_flat + noise
        else:
            scores_flat_noisy = scores_flat
        
        # [STEP 3] Top-K 选择
        keep_k = max(1, int(N_spatial * (1.0 - self.sparse_ratio)))
        # 使用 noisy score 进行排序选择
        _, top_indices = torch.topk(scores_flat_noisy, k=keep_k, dim=1)
        top_indices, _ = torch.sort(top_indices, dim=1)
        
        # [STEP 4] 特征收集
        x_flat = x.permute(0, 3, 4, 1, 2).reshape(B, N_spatial, T, C)
        # 扩展索引维度以匹配 [B, K, T, C]
        idx_expanded = top_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, T, C)
        x_gathered = torch.gather(x_flat, 1, idx_expanded)
        
        # [STEP 5] 梯度注入 (Additive STE)
        # 收集对应的 clean scores
        score_gathered = torch.gather(
            scores_flat.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, T, C), 
            1, idx_expanded
        )
        
        # 核心技巧：构造一个 Gate，数值为 1.0，但梯度来自 score
        # Forward: x_sparse = x_gathered * 1.0
        # Backward: grad 流向 x_gathered 和 score_gathered
        gate = 1.0 + (score_gathered - score_gathered.detach())
        x_sparse = x_gathered * gate
        
        return x_sparse, top_indices, (B, T, C, H, W)

    def densify(self, x_sparse, indices, original_shape):
        B, T, C, H, W = original_shape
        N_spatial = H * W
        x_dense_flat = torch.zeros(B, N_spatial, T, C, device=x_sparse.device, dtype=x_sparse.dtype)
        idx_expanded = indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, T, C)
        x_dense_flat.scatter_(1, idx_expanded, x_sparse)
        x_dense = x_dense_flat.view(B, H, W, T, C).permute(0, 3, 4, 1, 2)
        return x_dense

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

# ==========================================
# 增强型 Mamba 模块 (Enhanced Mamba Modules) - 保持不变
# ==========================================

class STMambaBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, 
                 d_state=16, d_conv=4, expand=2, use_checkpoint=False, 
                 sparse_ratio=0.0, 
                 anneal_start_epoch=5, anneal_end_epoch=10):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.target_sparse_ratio = sparse_ratio 
        self.anneal_start = anneal_start_epoch
        self.anneal_end = anneal_end_epoch
        
        self.norm1 = RMSNorm(dim)
        
        # Mamba 配置
        mamba_cfg = dict(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.scan = Mamba(**mamba_cfg)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = RMSNorm(dim)
        
        # --- 路径分支选择 ---
        if self.target_sparse_ratio > 0:
            self.sparse_handler = LearnableSparseHandler(dim, self.target_sparse_ratio)
            self.spatial_attn = nn.MultiheadAttention(
                embed_dim=dim, num_heads=4, batch_first=True, dropout=drop
            )
            self.norm_attn = nn.LayerNorm(dim)

        if self.target_sparse_ratio <= 0 or self.anneal_start > 0:
            self.mlp = LocalityEnhancedMLP(dim, int(dim * mlp_ratio), dim, act_layer=act_layer, drop=drop)
            
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)

    def _get_curr_sparse_ratio(self, epoch):
        if self.target_sparse_ratio <= 0: return 0.0
        if epoch < self.anneal_start: return 0.0
        if epoch >= self.anneal_end: return self.target_sparse_ratio
        progress = (epoch - self.anneal_start) / (self.anneal_end - self.anneal_start)
        return self.target_sparse_ratio * progress

    def forward(self, x_in, current_epoch=0):
        B, T, C, H, W = x_in.shape
        L = T * H * W
        
        curr_ratio = self._get_curr_sparse_ratio(current_epoch)
        shortcut = x_in
        
        # === 分支 A: 稀疏路径 ===
        if curr_ratio > 0 and hasattr(self, 'sparse_handler'):
            original_ratio = self.sparse_handler.sparse_ratio
            self.sparse_handler.sparse_ratio = curr_ratio
            
            x_sparse, indices, shape_info = self.sparse_handler.sparsify(x_in)
            self.sparse_handler.sparse_ratio = original_ratio
            
            K = x_sparse.shape[1]
            z_time = x_sparse.reshape(B * K, T, C)
            z_time = self.norm1(z_time)
            
            if self.use_checkpoint and z_time.requires_grad:
                z_time = checkpoint(self.scan, z_time, use_reentrant=False)
            else:
                z_time = self.scan(z_time)
                
            z_space_in = z_time.view(B, K, T, C).permute(0, 2, 1, 3).reshape(B * T, K, C)
            attn_out, _ = self.spatial_attn(z_space_in, z_space_in, z_space_in)
            z_space_out = self.norm_attn(z_space_in + attn_out)
            
            x_update_sparse = z_space_out.view(B, T, K, C).permute(0, 2, 1, 3) 
            x_update_dense = self.sparse_handler.densify(x_update_sparse, indices, shape_info)
            x_out = shortcut + self.drop_path(x_update_dense)
            
        # === 分支 B: 稠密路径 (Fallback/Warmup) ===
        else:
            x_norm = x_in.permute(0, 3, 4, 1, 2).reshape(B * H * W, T, C)
            x_norm = self.norm1(x_norm)
            
            if self.use_checkpoint and x_norm.requires_grad:
                z_time = checkpoint(self.scan, x_norm, use_reentrant=False)
            else:
                z_time = self.scan(x_norm)
                
            x_mamba = z_time.view(B, H, W, T, C).permute(0, 3, 4, 1, 2)
            x_mid = shortcut + self.drop_path(x_mamba)
            
            if hasattr(self, 'mlp'):
                x_mid_flat = x_mid.permute(0, 1, 3, 4, 2).reshape(B, L, C)
                x_mlp = self.mlp(self.norm2(x_mid_flat), H, W)
                x_out = x_mid + self.drop_path(x_mlp.view(B, T, H, W, C).permute(0, 1, 4, 2, 3))
            else:
                x_out = x_mid

        return x_out

class FlowPredictorGRU(nn.Module):
    def __init__(self, in_channels, hidden_dim):
        super().__init__()
        self.conv_z = nn.Conv2d(in_channels + hidden_dim, hidden_dim, 3, 1, 1)
        self.conv_r = nn.Conv2d(in_channels + hidden_dim, hidden_dim, 3, 1, 1)
        self.conv_h = nn.Conv2d(in_channels + hidden_dim, hidden_dim, 3, 1, 1)
        self.flow_head = nn.Conv2d(hidden_dim, 2, 3, 1, 1)
        self.residual_head = nn.Conv2d(hidden_dim, in_channels, 3, 1, 1)

        nn.init.constant_(self.flow_head.weight, 0)
        nn.init.constant_(self.flow_head.bias, 0)
        nn.init.constant_(self.residual_head.weight, 0)
        nn.init.constant_(self.residual_head.bias, 0)

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
    def __init__(self, dim, t_in, t_out):
        super().__init__()
        self.t_in = t_in
        self.t_out = t_out
        self.dim = dim
        self.flow_predictor = FlowPredictorGRU(in_channels=dim, hidden_dim=dim)
        self.init_h = nn.Conv2d(dim, dim, 1)

    def forward(self, z_seq):
        """
        [REFACTORED]
        输入现在是一个序列 z_seq，我们取其最后一帧作为平流的起点。
        """
        B, T_in, C, H, W = z_seq.shape
        # 取最后一帧作为当前状态
        curr_z = z_seq[:, -1]
        h = self.init_h(curr_z)
        
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H, device=z_seq.device),
            torch.linspace(-1, 1, W, device=z_seq.device),
            indexing='ij'
        )
        base_grid = torch.stack([xx, yy], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
        
        preds = []
        flow_maps = [] 
        
        for i in range(self.t_out):
            flow_delta, residual, h = self.flow_predictor(curr_z, h)
            flow_maps.append(flow_delta) 
            flow_field = flow_delta.permute(0, 2, 3, 1)
            grid = base_grid + flow_field
            z_advected = F.grid_sample(
                curr_z, grid, mode='bilinear', padding_mode='reflection', align_corners=False
            )
            # Advection + 自身的微小残差 (主要负责位移)
            next_z = z_advected + residual
            curr_z = next_z
            preds.append(curr_z)
            
        return torch.stack(preds, dim=1), torch.stack(flow_maps, dim=1)