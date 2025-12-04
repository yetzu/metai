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
# 稀疏计算组件 (Structured Sparse Components)
# ==========================================

class SpatialSparseTokenHandler(nn.Module):
    """
    [结构化空间稀疏处理器 - 重构版]
    策略：保持时间轴 T 完整，仅筛选高价值的空间坐标 (H, W)。
    物理意义：追踪活跃天气系统（如高反射率区域）的完整时间演变过程，忽略静止背景。
    """
    def __init__(self, sparse_ratio=0.5):
        super().__init__()
        self.sparse_ratio = sparse_ratio

    def sparsify(self, x):
        """
        Input: [B, T, C, H, W] (保持时空结构)
        
        Returns:
            x_sparse: [B, K, T, C]  <- K 是选中的空间点数，T 是完整时间序列
            indices:  [B, K]        <- 选中的空间索引 (0 ~ H*W-1)
            shape:    tuple         <- 用于恢复原始形状
        """
        B, T, C, H, W = x.shape
        N_spatial = H * W
        
        # 1. 计算空间重要性图 (Spatial Importance Map)
        # 策略：评估每个空间位置在整个时间段 T 内的总能量/活跃度
        # [B, T, C, H, W] -> [B, H, W] (对 T 和 C 求范数/平均)
        # 这里使用 L2 Norm 来表示能量
        spatial_energy = x.norm(dim=2).mean(dim=1) 
        
        # 展平空间维度: [B, H*W]
        spatial_scores = spatial_energy.view(B, N_spatial)
        
        # 2. 确定保留数量 K
        keep_k = max(1, int(N_spatial * (1.0 - self.sparse_ratio)))
        
        # 3. Top-K 选择 (仅在空间维度筛选)
        _, top_indices = torch.topk(spatial_scores, k=keep_k, dim=1)
        # 排序索引以保持空间相对顺序 (可选，但在纯时间 Mamba 中不影响 T 轴)
        top_indices, _ = torch.sort(top_indices, dim=1) # [B, K]
        
        # 4. Gather 操作 (提取完整的时空柱)
        # 目标: 从 [B, T, C, H, W] 中提取 [B, K, T, C]
        
        # 调整 x 形状以便 gather: [B, H*W, T, C]
        x_flat = x.permute(0, 3, 4, 1, 2).reshape(B, N_spatial, T, C)
        
        # 扩展索引以匹配 T 和 C 维度
        # indices: [B, K] -> [B, K, T, C]
        idx_expanded = top_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, T, C)
        
        # 提取: [B, K, T, C]
        x_sparse = torch.gather(x_flat, 1, idx_expanded)
        
        return x_sparse, top_indices, (B, T, C, H, W)

    def densify(self, x_sparse, indices, original_shape):
        """
        将稀疏计算结果填回原始稠密形状。
        Args:
            x_sparse: [B, K, T, C]
            indices:  [B, K]
            original_shape: (B, T, C, H, W)
        Returns:
            x_dense: [B, T, C, H, W] (未选中位置填充为 0)
        """
        B, T, C, H, W = original_shape
        K = x_sparse.shape[1]
        N_spatial = H * W
        
        # 准备空白容器: [B, H*W, T, C]
        x_dense_flat = torch.zeros(B, N_spatial, T, C, device=x_sparse.device, dtype=x_sparse.dtype)
        
        # 扩展索引
        idx_expanded = indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, T, C)
        
        # 填充回原位
        x_dense_flat.scatter_(1, idx_expanded, x_sparse)
        
        # 恢复原始形状: [B, H*W, T, C] -> [B, T, C, H, W]
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
# 增强型 Mamba 模块 (Enhanced Mamba Modules)
# ==========================================

class STMambaBlock(nn.Module):
    """
    [结构化稀疏 Mamba 模块 - 重构版]
    架构：Input -> Spatial Sparsify -> Time-Domain Mamba -> Spatial Mixing -> Densify -> Output
    
    核心改进：
    1. 使用 'SpatialSparseTokenHandler'，避免破坏时间因果性。
    2. Mamba 扫描严格限制在时间轴 T 上，每个空间点视为独立的序列。
    3. 引入 'spatial_mixer' 在稀疏状态下进行空间信息交互。
    """
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, 
                 d_state=16, d_conv=4, expand=2, use_checkpoint=False, 
                 sparse_ratio=0.0, 
                 anneal_start_epoch=5, anneal_end_epoch=10):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.target_sparse_ratio = sparse_ratio # 目标稀疏率
        self.anneal_start = anneal_start_epoch
        self.anneal_end = anneal_end_epoch
        
        self.norm1 = RMSNorm(dim)
        
        # --- Mamba 配置 ---
        # 关键修改：Mamba 现在处理的是单一空间点的"时间序列"
        # 形状预期: [Batch_Size * K, Seq_Len=T, Dim]
        mamba_cfg = dict(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.scan = Mamba(**mamba_cfg)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = RMSNorm(dim)
        
        # --- 稀疏/混合组件 ---
        # 初始化时只要 target > 0 就实例化稀疏组件
        if self.target_sparse_ratio > 0:
            self.sparse_handler = SpatialSparseTokenHandler(self.target_sparse_ratio)
            # 空间混合层 (Spatial Mixing): 允许被选中的 K 个点之间交换信息
            # 简单的 MLP 保持高效
            self.spatial_mixer = nn.Sequential(
                nn.Linear(dim, dim),
                nn.GELU(),
                nn.Linear(dim, dim)
            )
        else:
            # 稠密模式下的 MLP
            self.mlp = LocalityEnhancedMLP(dim, int(dim * mlp_ratio), dim, act_layer=act_layer, drop=drop)
            
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)

    def _get_curr_sparse_ratio(self, epoch):
        """计算当前的退火稀疏率"""
        if self.target_sparse_ratio <= 0: return 0.0
        if epoch < self.anneal_start: return 0.0
        if epoch >= self.anneal_end: return self.target_sparse_ratio
        
        # 线性插值
        progress = (epoch - self.anneal_start) / (self.anneal_end - self.anneal_start)
        return self.target_sparse_ratio * progress

    def forward(self, x_in, current_epoch=0):
        """
        Args:
            x_in: [B, T, C, H, W]
            current_epoch: 当前训练轮次，用于计算稀疏率
        """
        B, T, C, H, W = x_in.shape
        L = T * H * W
        
        # 动态计算当前稀疏率
        curr_ratio = self._get_curr_sparse_ratio(current_epoch)
        
        shortcut = x_in
        
        # === 分支 A: 结构化稀疏路径 (High Performance & Physics Aware) ===
        # 仅当当前比率 > 0 且拥有稀疏组件时启用
        if curr_ratio > 0 and hasattr(self, 'sparse_handler'):
            # 临时覆盖 handler 的 ratio
            original_ratio = self.sparse_handler.sparse_ratio
            self.sparse_handler.sparse_ratio = curr_ratio
            
            # 1. 稀疏化: [B, K, T, C]
            # 此时保持了 T 的完整性，K 是活跃的空间点
            # x_norm = x_in # 在 sparse_handler 内部会计算 norm
            x_sparse, indices, shape_info = self.sparse_handler.sparsify(x_in)
            
            # 恢复 ratio 状态
            self.sparse_handler.sparse_ratio = original_ratio
            
            K = x_sparse.shape[1]
            
            # 2. 准备 Mamba 输入
            # 将 (B, K) 合并，视作独立的 Batch，序列长度为 T
            # [B, K, T, C] -> [B*K, T, C]
            z_time = x_sparse.reshape(B * K, T, C)
            z_time = self.norm1(z_time)
            
            # 3. Mamba 时间扫描 (Time Evolution)
            # 仅在时间轴上进行因果推演
            if self.use_checkpoint and z_time.requires_grad:
                z_time = checkpoint(self.scan, z_time, use_reentrant=False)
            else:
                z_time = self.scan(z_time)
                
            # 4. 空间混合 (Spatial Mixing)
            # Mamba 完成后，让这 K 个点交互 (模拟平流/扩散)
            # [B*K, T, C] -> [B, T, K, C]
            z_space = z_time.view(B, K, T, C).permute(0, 2, 1, 3) 
            z_space = self.norm2(z_space)
            z_space = self.spatial_mixer(z_space) # MLP 在 C 维度操作，但在 K 维度共享权重
            
            # 5. 残差连接 + 反稀疏化
            # 恢复形状 [B, K, T, C]
            x_update_sparse = z_space.permute(0, 2, 1, 3) 
            
            # 将更新量加回 (Dense shortcut + Sparse update)
            # 未被选中的点 update 为 0
            x_update_dense = self.sparse_handler.densify(x_update_sparse, indices, shape_info)
            
            # 最终输出
            x_out = shortcut + self.drop_path(x_update_dense)
            
        # === 分支 B: 稠密路径 (Fallback/Warmup) ===
        else:
            # 兼容 Mamba 扫描逻辑，但全量计算
            # 展平以便统一处理：这里为了利用 Mamba 的序列特性，我们将 H*W 视为 Batch
            # [B, T, C, H, W] -> [B, H*W, T, C] -> [B*H*W, T, C]
            x_norm = x_in.permute(0, 3, 4, 1, 2).reshape(B * H * W, T, C)
            x_norm = self.norm1(x_norm)
            
            # 时间扫描
            if self.use_checkpoint and x_norm.requires_grad:
                z_time = checkpoint(self.scan, x_norm, use_reentrant=False)
            else:
                z_time = self.scan(x_norm)
                
            # 还原形状
            # [B*H*W, T, C] -> [B, T, C, H, W]
            x_mamba = z_time.view(B, H, W, T, C).permute(0, 3, 4, 1, 2)
            x_mid = shortcut + self.drop_path(x_mamba)
            
            # 空间混合 (使用全卷积/MLP)
            # 此时使用 LocalityEnhancedMLP (带 DWConv)
            # 需要适配 MLP 输入 [B, L, C] -> [B, T*H*W, C]
            x_mid_flat = x_mid.permute(0, 1, 3, 4, 2).reshape(B, L, C)
            x_mlp = self.mlp(self.norm2(x_mid_flat), H, W)
            
            x_out = x_mid + self.drop_path(x_mlp.view(B, T, H, W, C).permute(0, 1, 4, 2, 3))

        return x_out

# ==========================================
# 增强型时间投影组件 (Advective Projection)
# ==========================================

class FlowPredictorGRU(nn.Module):
    """
    [动态流场预测器 - GRU版]
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
        
        return delta_flow, h_next

class AdvectiveProjection(nn.Module):
    """
    [显式平流投影层]
    利用物理约束 (流场) 对特征进行时间推进。
    
    修改：
    padding_mode='zeros'，避免物理上的"镜像反射"违背，更符合开放边界系统。
    """
    def __init__(self, dim, t_in, t_out):
        super().__init__()
        self.t_in = t_in
        self.t_out = t_out
        self.dim = dim
        self.flow_predictor = FlowPredictorGRU(in_channels=dim, hidden_dim=dim)
        self.init_h = nn.Conv2d(dim, dim, 1)

    def forward(self, z_seq):
        B, T_in, C, H, W = z_seq.shape
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
            flow_delta, h = self.flow_predictor(curr_z, h)
            
            # 收集流场增量
            flow_maps.append(flow_delta) 
            
            flow_field = flow_delta.permute(0, 2, 3, 1)
            grid = base_grid + flow_field
            
            # [物理修正] 使用 'zeros' 填充，避免边界产生虚假回波
            next_z = F.grid_sample(curr_z, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
            
            curr_z = next_z
            preds.append(curr_z)
            
        return torch.stack(preds, dim=1), torch.stack(flow_maps, dim=1)