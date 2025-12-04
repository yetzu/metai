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
# (RMSNorm, BasicConv2d, ConvSC, ResizeConv, TimeAlignBlock 代码保持原样，此处省略以节省篇幅)

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
        x = x.permute(0, 2, 3, 4, 1).contiguous() 
        x = self.time_proj(x) 
        x = x.permute(0, 4, 1, 2, 3).contiguous() 
        B, T, C, H, W = x.shape
        x = x.view(B, T, C, -1).permute(0, 1, 3, 2)
        x = self.norm(x)
        x = x.permute(0, 1, 3, 2).view(B, T, C, H, W)
        return x

# ==========================================
# 稀疏计算组件 (Sparse Computation Components)
# ==========================================

class SparseTokenHandler(nn.Module):
    """
    [稀疏 Token 处理器 - 改进版]
    实现真正的计算稀疏化，并支持“条件相关性”采样。
    """
    def __init__(self, sparse_ratio=0.5, random_ratio=0.1):
        super().__init__()
        self.sparse_ratio = sparse_ratio
        self.random_ratio = random_ratio

    def sparsify(self, x, importance_score=None):
        """
        Dense [B, L, C] -> Sparse [B, K, C]
        
        Args:
            x: 输入特征
            importance_score: [B, L] 外部计算的重要性分数 (可选)。
                              若提供，则基于此分数进行 Top-K 选择；否则基于 L2 Norm。
        """
        B, L, C = x.shape
        
        # [逻辑变更] 优先使用外部传入的重要性分数 (如基于时序差分的)
        if importance_score is not None:
            # 确保维度匹配 [B, L]
            if importance_score.dim() == 3: 
                score = importance_score.mean(dim=-1)
            else:
                score = importance_score
        else:
            # 默认使用 L2 Norm
            score = x.norm(dim=-1)
        
        # 确定保留数量 K
        keep_len = max(1, int(L * (1.0 - self.sparse_ratio)))
        
        # 混合采样策略: Top-K (强信号) + Random (弱信号探查)
        if self.random_ratio > 0 and keep_len > 1:
            num_random = int(keep_len * self.random_ratio)
            num_top = max(1, keep_len - num_random)
            num_random = max(0, keep_len - num_top)
        else:
            num_top = keep_len
            num_random = 0
            
        # 1. 获取 Top-K 索引 (强信号)
        _, top_indices = torch.topk(score, k=num_top, dim=1)
        
        # 2. 获取随机索引 (弱信号)
        if num_random > 0:
            mask = torch.ones(B, L, device=x.device, dtype=torch.bool)
            mask.scatter_(1, top_indices, False)
            
            rand_indices_list = []
            for b in range(B):
                remaining_indices = torch.nonzero(mask[b], as_tuple=False).squeeze(-1)
                if len(remaining_indices) >= num_random:
                    perm = torch.randperm(len(remaining_indices), device=x.device)[:num_random]
                    chosen = remaining_indices[perm]
                else:
                    chosen = remaining_indices
                rand_indices_list.append(chosen)
            
            rand_indices = torch.stack(rand_indices_list, dim=0)
            indices = torch.cat([top_indices, rand_indices], dim=1)
        else:
            indices = top_indices
        
        # 排序索引保持时序/空间相对顺序
        indices, _ = torch.sort(indices, dim=1)
        
        # Gather 提取 Active Tokens
        batch_indices = torch.arange(B, device=x.device).unsqueeze(1).expand(-1, indices.shape[1])
        x_sparse = x[batch_indices, indices]
        
        return x_sparse, indices, (B, L, C)

    def densify(self, x_sparse, indices, original_shape):
        """Sparse [B, K, C] -> Dense [B, L, C]"""
        B, L, C = original_shape
        keep_len = x_sparse.shape[1]
        x_dense = torch.zeros(B, L, C, device=x_sparse.device, dtype=x_sparse.dtype)
        batch_indices = torch.arange(B, device=x_sparse.device).unsqueeze(1).expand(-1, keep_len)
        x_dense[batch_indices, indices] = x_sparse
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

class SparseLocalityEnhancedMLP(nn.Module):
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
    def forward(self, x_sparse, indices, B, L, H, W):
        K = x_sparse.shape[1]
        x = self.fc1(x_sparse)
        x_dense = torch.zeros(B, L, x.shape[-1], device=x.device, dtype=x.dtype)
        batch_indices = torch.arange(B, device=x.device).unsqueeze(1).expand(-1, K)
        x_dense[batch_indices, indices] = x
        T = L // (H * W)
        x_spatial = x_dense.view(B * T, H, W, -1).permute(0, 3, 1, 2)
        x_spatial = self.dwconv(x_spatial)
        x_flat = x_spatial.permute(0, 2, 3, 1).reshape(B, L, -1)
        x_sparse_new = x_flat[batch_indices, indices] 
        x_sparse_new = self.act(x_sparse_new)
        x_sparse_new = self.drop(x_sparse_new)
        x = self.fc2(x_sparse_new)
        x = self.drop(x)
        return x

# ==========================================
# 增强型 Mamba 模块 (Enhanced Mamba Modules)
# ==========================================

class STMambaBlock(nn.Module):
    """
    [稀疏感知时空 Mamba 模块]
    集成 "Sandwich" 架构，并利用时序差分进行条件采样。
    """
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, 
                 d_state=16, d_conv=4, expand=2, use_checkpoint=False, 
                 sparse_ratio=0.0):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.sparse_ratio = sparse_ratio
        
        self.norm1 = RMSNorm(dim)
        
        mamba_cfg = dict(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.scan = Mamba(**mamba_cfg)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = RMSNorm(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        
        if self.sparse_ratio > 0:
            self.mlp = SparseLocalityEnhancedMLP(dim, mlp_hidden_dim, dim, act_layer=act_layer, drop=drop)
            self.sparse_handler = SparseTokenHandler(sparse_ratio)
        else:
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
        
        # === [新增] 计算时空差分重要性图 ===
        importance = None
        if self.sparse_ratio > 0:
            # 计算时序差分: diff[t] = |x[t] - x[t-1]|
            # 第0帧使用自身代替
            x_diff = torch.zeros_like(x_in)
            x_diff[:, 1:] = x_in[:, 1:] - x_in[:, :-1]
            x_diff[:, 0] = x_in[:, 0] 
            
            # 展平以便计算 Norm
            flat_x = x_in.permute(0, 1, 3, 4, 2).reshape(B, L, C)
            flat_diff = x_diff.permute(0, 1, 3, 4, 2).reshape(B, L, C)
            
            # 融合重要性: alpha * 绝对强度 + beta * 变化强度
            # beta > alpha 强调变化剧烈的区域（如初生对流）
            alpha, beta = 1.0, 2.0 
            importance = alpha * flat_x.norm(dim=-1) + beta * flat_diff.norm(dim=-1)
            
            x = flat_x
        else:
            x = x_in.permute(0, 1, 3, 4, 2).reshape(B, L, C)
        
        shortcut = x
        x_norm = self.norm1(x)

        # === 稀疏计算路径 ===
        if self.sparse_ratio > 0:
            # [修改] 传入基于差分计算的 importance
            x_sparse, indices, _ = self.sparse_handler.sparsify(x_norm, importance_score=importance)
            
            # 稀疏 Mamba 扫描
            def sparse_scan(z):
                return self.scan(z) + self.scan(z.flip([1])).flip([1])

            if self.use_checkpoint and x_sparse.requires_grad:
                mamba_out_sparse = checkpoint(sparse_scan, x_sparse, use_reentrant=False)
            else:
                mamba_out_sparse = sparse_scan(x_sparse)
            
            # 稀疏 MLP
            batch_indices = torch.arange(B, device=x.device).unsqueeze(1).expand(-1, indices.shape[1])
            x_val_at_indices = x[batch_indices, indices]
            mid_sparse = x_val_at_indices + self.drop_path(mamba_out_sparse)
            mlp_in_sparse = self.norm2(mid_sparse)
            mlp_out_sparse = self.mlp(mlp_in_sparse, indices, B, L, H, W)
            
            total_sparse_update = self.drop_path(mamba_out_sparse) + self.drop_path(mlp_out_sparse)
            
            # 统一还原 (Densify)
            update_dense = self.sparse_handler.densify(total_sparse_update, indices, (B, L, C))
            x = shortcut + update_dense
            
        else:
            # === 稠密计算路径 ===
            def scan_multidir(z_row, B, T, H, W, C):
                L = T * H * W
                out_h = self.scan(z_row) + self.scan(z_row.flip([1])).flip([1])
                z_col = z_row.view(B, T, H, W, C).permute(0, 1, 3, 2, 4).contiguous().reshape(B, L, C)
                out_v = self.scan(z_col) + self.scan(z_col.flip([1])).flip([1])
                out_v = out_v.view(B, T, W, H, C).permute(0, 1, 3, 2, 4).contiguous().reshape(B, L, C)
                return out_h + out_v

            if self.use_checkpoint and x.requires_grad:
                out = checkpoint(scan_multidir, x_norm, B, T, H, W, C, use_reentrant=False)
            else:
                out = scan_multidir(x_norm, B, T, H, W, C)

            x = shortcut + self.drop_path(out)
            x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        
        return x.reshape(B, T, H, W, C).permute(0, 1, 4, 2, 3).contiguous()

# ==========================================
# 增强型时间投影组件 (Advective Projection - Final)
# ==========================================

class FlowPredictorGRU(nn.Module):
    """
    [动态流场预测器 - GRU版 - 修复版]
    利用 ConvGRU 单元动态预测流场增量，并维护隐藏状态。
    """
    def __init__(self, in_channels, hidden_dim):
        super().__init__()
        self.conv_z = nn.Conv2d(in_channels + hidden_dim, hidden_dim, 3, 1, 1)
        self.conv_r = nn.Conv2d(in_channels + hidden_dim, hidden_dim, 3, 1, 1)
        self.conv_h = nn.Conv2d(in_channels + hidden_dim, hidden_dim, 3, 1, 1)
        
        # 流场输出头: 预测流场 [dx, dy]
        self.flow_head = nn.Conv2d(hidden_dim, 2, 3, 1, 1)
        # 初始化为0，保证初始状态平稳
        nn.init.constant_(self.flow_head.weight, 0)
        nn.init.constant_(self.flow_head.bias, 0)

    def forward(self, x, h):
        """
        x: 当前 Warp 后的特征 [B, C, H, W]
        h: 上一时刻的隐状态 [B, C_hid, H, W]
        """
        combined = torch.cat([x, h], dim=1)
        z = torch.sigmoid(self.conv_z(combined))
        r = torch.sigmoid(self.conv_r(combined))
        
        combined_new = torch.cat([x, r * h], dim=1)
        h_tilde = torch.tanh(self.conv_h(combined_new))
        
        h_next = (1 - z) * h + z * h_tilde
        delta_flow = self.flow_head(h_next)
        
        # [关键修复] 返回更新后的隐藏状态 h_next
        return delta_flow, h_next

class AdvectiveProjection(nn.Module):
    """
    [显式平流投影层 - 真实 Warping 实现]
    利用 GRU 预测流场，并使用 Grid Sample 对特征图进行物理平流推演。
    替代了原本缺失的实现。
    """
    def __init__(self, dim, t_in, t_out):
        super().__init__()
        self.t_in = t_in
        self.t_out = t_out
        self.dim = dim
        
        # GRU 预测器: 输入特征维度与 hidden 维度相同
        self.flow_predictor = FlowPredictorGRU(in_channels=dim, hidden_dim=dim)
        
        # 初始化 GRU 隐藏状态的投影层
        self.init_h = nn.Conv2d(dim, dim, 1)

    def forward(self, z_seq):
        """
        z_seq: [B, T_in, C, H, W] 观测阶段的特征序列
        返回: [B, T_out, C, H, W] 预测阶段的特征序列
        """
        B, T_in, C, H, W = z_seq.shape
        
        # 1. 以最后一帧观测特征作为推演的起始点
        curr_z = z_seq[:, -1] # [B, C, H, W]
        
        # 2. 初始化 GRU 隐藏状态
        h = self.init_h(curr_z)
        
        # 3. 预计算归一化坐标网格 (Base Grid)
        # 范围 [-1, 1], 形状 [B, H, W, 2]
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H, device=z_seq.device),
            torch.linspace(-1, 1, W, device=z_seq.device),
            indexing='ij'
        )
        base_grid = torch.stack([xx, yy], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
        
        preds = []
        
        # 4. 循环生成 T_out 帧
        for i in range(self.t_out):
            # A. 预测流场 (Flow)
            # flow_delta: [B, 2, H, W], 代表归一化坐标下的 (dx, dy)
            # h: 更新后的状态
            flow_delta, h = self.flow_predictor(curr_z, h)
            
            # B. 构建采样网格
            # flow_delta permute -> [B, H, W, 2]
            flow_field = flow_delta.permute(0, 2, 3, 1)
            
            # 这里的 flow_delta 应当通过 Tanh 或类似机制约束范围，
            # 这里假设 flow_predictor 输出已经是适配 grid_sample 的尺度。
            # 真实场景中，流场叠加 grid:
            grid = base_grid + flow_field
            
            # C. 执行平流 (Warping)
            # 使用双线性插值采样，padding_mode='border' 防止边界黑边引入非物理梯度
            next_z = F.grid_sample(curr_z, grid, mode='bilinear', padding_mode='border', align_corners=False)
            
            # D. 更新当前帧 (此处可加入额外的残差卷积来模拟生消，此处保持纯平流)
            curr_z = next_z
            
            preds.append(curr_z)
            
        return torch.stack(preds, dim=1)