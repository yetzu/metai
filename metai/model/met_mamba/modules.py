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
        x = x.permute(0, 2, 3, 4, 1).contiguous() 
        x = self.time_proj(x) 
        x = x.permute(0, 4, 1, 2, 3).contiguous() 
        B, T, C, H, W = x.shape
        x = x.view(B, T, C, -1).permute(0, 1, 3, 2)
        x = self.norm(x)
        x = x.permute(0, 1, 3, 2).view(B, T, C, H, W)
        return x

# ==========================================
# 稀疏计算组件 (Sparse Computation Components) [新增]
# ==========================================

class SparseTokenHandler(nn.Module):
    """
    [稀疏 Token 处理器]
    实现真正的计算稀疏化：只对 Top-K 的活跃 Token 进行计算。
    """
    def __init__(self, sparse_ratio=0.5):
        super().__init__()
        self.sparse_ratio = sparse_ratio

    def sparsify(self, x):
        """
        Dense [B, L, C] -> Sparse [B, K, C]
        返回: x_sparse, indices (用于还原), B, L, C
        """
        B, L, C = x.shape
        # 计算 Token 重要性 (L2 Norm)
        # x_norm: [B, L]
        x_norm = x.norm(dim=-1)
        
        # 确定保留的 Token 数量 K
        # 至少保留 1 个 token 避免崩溃
        keep_len = max(1, int(L * (1.0 - self.sparse_ratio)))
        
        # 获取 Top-K 索引
        # indices: [B, K]
        _, indices = torch.topk(x_norm, k=keep_len, dim=1)
        
        # 排序索引，尽量保持原有的相对顺序 (这对 Mamba 的时序性有帮助)
        indices, _ = torch.sort(indices, dim=1)
        
        # Gather 提取 Active Tokens
        # batch_indices: [B, K] -> [[0,0,..], [1,1,..]]
        batch_indices = torch.arange(B, device=x.device).unsqueeze(1).expand(-1, keep_len)
        
        # x_sparse: [B, K, C]
        x_sparse = x[batch_indices, indices]
        
        return x_sparse, indices, (B, L, C)

    def densify(self, x_sparse, indices, original_shape):
        """
        Sparse [B, K, C] -> Dense [B, L, C] (未选中位置填 0)
        """
        B, L, C = original_shape
        keep_len = x_sparse.shape[1]
        
        # 创建全 0 画布
        x_dense = torch.zeros(B, L, C, device=x_sparse.device, dtype=x_sparse.dtype)
        
        # Scatter 回填
        batch_indices = torch.arange(B, device=x_sparse.device).unsqueeze(1).expand(-1, keep_len)
        x_dense[batch_indices, indices] = x_sparse
        
        return x_dense

# ==========================================
# 增强型 Mamba 模块 (Enhanced Mamba Modules)
# ==========================================

class STMambaBlock(nn.Module):
    """
    [稀疏感知时空 Mamba 模块]
    支持 Top-K Token Pruning，实现真正的 FLOPs 减少。
    """
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, 
                 d_state=16, d_conv=4, expand=2, use_checkpoint=False, 
                 sparse_ratio=0.0): # 新增 sparse_ratio
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.sparse_ratio = sparse_ratio
        
        self.norm1 = RMSNorm(dim)
        
        mamba_cfg = dict(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.scan = Mamba(**mamba_cfg)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = RMSNorm(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = LocalityEnhancedMLP(dim, mlp_hidden_dim, dim, act_layer=act_layer, drop=drop)
        
        # 初始化稀疏处理器
        if self.sparse_ratio > 0:
            self.sparse_handler = SparseTokenHandler(sparse_ratio)
            
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, x_in):
        # x_in: [B, T, C, H, W]
        B, T, C, H, W = x_in.shape
        L = T * H * W
        
        # 展平: [B, L, C] (这里我们将 Batch 和 Time 维度合并考虑还是分开？
        # 为了兼容 Mamba 的输入 [Batch, SeqLen, Dim]，通常是将 B*T 合并，或者 B 作为 Batch。
        # 原代码是 x_in.permute(...).reshape(B, L, C)，即 Batch=B, Seq=T*H*W。
        # 这样能捕捉 T 维度的依赖。
        x = x_in.permute(0, 1, 3, 4, 2).contiguous().reshape(B, L, C)
        
        shortcut = x
        x_norm = self.norm1(x)

        # === 稀疏计算逻辑 ===
        if self.sparse_ratio > 0:
            # 1. 剪枝: 仅保留活跃 Token
            # x_sparse: [B, K, C]
            x_sparse, indices, _ = self.sparse_handler.sparsify(x_norm)
            
            # 2. 稀疏 Mamba 扫描
            # 注意：在稀疏模式下，我们暂时放弃"垂直扫描"，因为网格结构已被破坏。
            # 但由于 Token 是按顺序 gather 的 (sort indices)，保留了相对位置关系。
            # Mamba 的长程能力可以弥补 2D 结构的丢失。
            # 我们做双向扫描 (Forward + Backward)
            def sparse_scan(z):
                return self.scan(z) + self.scan(z.flip([1])).flip([1])

            if self.use_checkpoint and x_sparse.requires_grad:
                out_sparse = checkpoint(sparse_scan, x_sparse, use_reentrant=False)
            else:
                out_sparse = sparse_scan(x_sparse)
            
            # 3. 还原: 填回 Dense Tensor (未选中区域为 0)
            out = self.sparse_handler.densify(out_sparse, indices, (B, L, C))
            
        else:
            # === 原有 Dense 逻辑 (包含垂直扫描) ===
            def scan_multidir(z_row, B, T, H, W, C):
                L = T * H * W
                # 水平
                out_h = self.scan(z_row) + self.scan(z_row.flip([1])).flip([1])
                # 垂直
                z_col = z_row.view(B, T, H, W, C).permute(0, 1, 3, 2, 4).contiguous().reshape(B, L, C)
                out_v = self.scan(z_col) + self.scan(z_col.flip([1])).flip([1])
                out_v = out_v.view(B, T, W, H, C).permute(0, 1, 3, 2, 4).contiguous().reshape(B, L, C)
                return out_h + out_v

            if self.use_checkpoint and x.requires_grad:
                out = checkpoint(scan_multidir, x_norm, B, T, H, W, C, use_reentrant=False)
            else:
                out = scan_multidir(x_norm, B, T, H, W, C)

        # DropPath & Residual
        x = shortcut + self.drop_path(out)
        
        # MLP (Locality Enhanced)
        # 注意：MLP 依然在 Dense 模式下运行，因为它包含 DWConv2d，需要网格结构。
        # 虽然这部分有计算，但 Conv 开销远小于 Mamba。
        # 且由于输入大部分是 0 (sparse 模式下)，实际有效信息流依然聚焦在云团。
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        
        return x.reshape(B, T, H, W, C).permute(0, 1, 4, 2, 3).contiguous()

# ==========================================
# 增强型时间投影组件 (Advective Projection - Final)
# ==========================================

class FlowPredictorGRU(nn.Module):
    def __init__(self, in_channels, hidden_dim):
        super().__init__()
        self.conv_z = nn.Conv2d(in_channels + hidden_dim, hidden_dim, 3, 1, 1)
        self.conv_r = nn.Conv2d(in_channels + hidden_dim, hidden_dim, 3, 1, 1)
        self.conv_h = nn.Conv2d(in_channels + hidden_dim, hidden_dim, 3, 1, 1)
        self.flow_head = nn.Conv2d(hidden_dim, 2, 3, 1, 1)
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
        return delta_flow, h_next

def warp(x, flow):
    B, C, H, W = x.shape
    if not hasattr(warp, 'grid') or warp.grid.shape != (1, H, W, 2) or warp.grid.device != x.device:
        xx = torch.arange(0, W, device=x.device).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H, device=x.device).view(-1, 1).repeat(1, W)
        warp.grid = torch.stack((xx, yy), dim=-1).float().unsqueeze(0)
    
    grid = warp.grid.repeat(B, 1, 1, 1)
    flow_perm = flow.permute(0, 2, 3, 1)
    vgrid = grid + flow_perm
    vgrid[:, :, :, 0] = 2.0 * vgrid[:, :, :, 0] / max(W - 1, 1) - 1.0
    vgrid[:, :, :, 1] = 2.0 * vgrid[:, :, :, 1] / max(H - 1, 1) - 1.0
    output = F.grid_sample(x, vgrid, mode='bilinear', padding_mode='border', align_corners=True)
    return output

class SparsityGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 3, 1, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(dim // 4, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.gate(x)

class AdvectiveProjection(nn.Module):
    def __init__(self, dim, t_in, t_out):
        super().__init__()
        self.t_in = t_in
        self.t_out = t_out
        self.dim = dim
        self.init_encoder = nn.Sequential(
            nn.Conv2d(dim * t_in, dim, 1), 
            nn.SiLU(inplace=True),
            nn.Conv2d(dim, dim, 3, 1, 1) 
        )
        self.flow_gru = FlowPredictorGRU(in_channels=dim, hidden_dim=dim)
        self.sparsity_gate = SparsityGate(dim)
        self.evolution_net = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=3, padding=1),
            nn.GroupNorm(4, dim),
            nn.SiLU(inplace=True),
            nn.Conv3d(dim, dim, kernel_size=3, padding=1),
            nn.GroupNorm(4, dim),
            nn.SiLU(inplace=True)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, x):
        B, T_in, C, H, W = x.shape
        x_flat = x.view(B, T_in * C, H, W)
        h_t = self.init_encoder(x_flat) 
        curr_feat = x[:, -1]
        curr_flow = torch.zeros(B, 2, H, W, device=x.device)
        warped_preds = []
        
        for t in range(self.t_out):
            delta_flow, h_t = self.flow_gru(curr_feat, h_t)
            curr_flow = curr_flow + delta_flow
            curr_feat_warped = warp(curr_feat, curr_flow)
            importance_mask = self.sparsity_gate(curr_feat_warped)
            curr_feat = curr_feat_warped * importance_mask
            warped_preds.append(curr_feat)
            
        x_warped = torch.stack(warped_preds, dim=1)
        x_evolution = x.permute(0, 2, 1, 3, 4)
        if T_in != self.t_out:
            x_evolution = F.interpolate(x_evolution, size=(self.t_out, H, W), mode='trilinear', align_corners=False)
        x_resid = self.evolution_net(x_evolution)
        x_resid = x_resid.permute(0, 2, 1, 3, 4)
        return x_warped + x_resid