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

class LearnableSparseHandler(nn.Module):
    """
    [改进版: 可微稀疏处理器]
    
    核心机制:
    1. Router Network: 使用轻量级 CNN 预测每个空间位置的重要性得分 (Score)。
    2. Straight-Through Estimator (STE): 
       在 Forward 阶段使用硬截断 (Top-K) 提取特征;
       在 Backward 阶段通过重参数化技巧允许梯度流回 Score 网络。
       
    [v2.1 改进]: 引入噪声注入 (Noise Injection) 和 Clean Gradient Scaling 以增强训练稳定性。
    """
    def __init__(self, dim, sparse_ratio=0.5):
        super().__init__()
        self.sparse_ratio = sparse_ratio
        
        # 路由网络：从特征中提取空间重要性热力图
        # 输入 dim -> 降维 -> 1通道 Score
        self.router = nn.Sequential(
            nn.Conv2d(dim, dim // 4, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(dim // 4, 1, kernel_size=1)
        )
        
        # [新增] 注册一个极小的扰动系数，防止除零错误和打破平局
        self.register_buffer('noise_scale', torch.tensor(0.01))
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)

    def sparsify(self, x):
        """
        Input: [B, T, C, H, W]
        
        Returns:
            x_sparse: [B, K, T, C]  (包含梯度注入)
            top_indices: [B, K]     (用于恢复位置)
            shape_info: tuple       (原始形状)
        """
        B, T, C, H, W = x.shape
        N_spatial = H * W
        
        # 1. 计算重要性分数 (Learnable Score)
        # 策略：计算时间维度的均值作为空间分布的代表 (假设短时间内活跃区域相对稳定)
        # x_mean: [B, C, H, W]
        x_mean = x.mean(dim=1) 
        
        # 生成 Score Map: [B, 1, H, W] -> [B, H*W]
        scores_map = torch.sigmoid(self.router(x_mean)) 
        scores_flat = scores_map.view(B, N_spatial)
        
        # [增强稳定性 1] 训练阶段注入随机噪声 (Noise Injection)
        # 这有助于打破早期训练的对称性，避免 Router 坍缩到固定点
        if self.training:
            noise = torch.randn_like(scores_flat) * self.noise_scale
            scores_flat_noisy = scores_flat + noise
        else:
            scores_flat_noisy = scores_flat
        
        # 2. 确定保留数量 K
        keep_k = max(1, int(N_spatial * (1.0 - self.sparse_ratio)))
        
        # 3. Top-K 选择 (Forward Pass)
        # 使用加噪后的分数进行排序
        top_scores, top_indices = torch.topk(scores_flat_noisy, k=keep_k, dim=1)
        
        # 排序索引以保持空间相对顺序 (这对某些位置敏感的操作有帮助)
        top_indices, _ = torch.sort(top_indices, dim=1) # [B, K]
        
        # 4. 特征提取 (Gather)
        # 调整 x 形状: [B, H*W, T, C]
        x_flat = x.permute(0, 3, 4, 1, 2).reshape(B, N_spatial, T, C)
        
        # 扩展索引维度以匹配特征: [B, K] -> [B, K, T, C]
        idx_expanded = top_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, T, C)
        
        # 提取特征
        x_gathered = torch.gather(x_flat, 1, idx_expanded)
        
        # 5. [关键] 梯度注入 (Gradient Injection via STE)
        # [增强稳定性 2] 使用原始(无噪)的 score 计算梯度权重
        # 确保梯度流向真实的 Router 输出，而不是拟合噪声
        
        # 重新获取对应位置的 原始 Score，并扩展维度
        score_gathered_clean = torch.gather(
            scores_flat.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, T, C), 
            1, idx_expanded
        )
        
        # 构造 STE:
        # Forward: x_sparse = x_gathered * 1.0 (数值不变)
        # Backward: x_sparse = x_gathered * score (梯度流经 score)
        # 实现方式: x * (s / s.detach())
        # 加入 eps 防止数值不稳定
        eps = 1e-6
        scale_factor = score_gathered_clean / (score_gathered_clean.detach() + eps)
        
        x_sparse = x_gathered * scale_factor
        
        return x_sparse, top_indices, (B, T, C, H, W)

    def densify(self, x_sparse, indices, original_shape):
        """
        将稀疏 Token 填回原始稠密网格。
        """
        B, T, C, H, W = original_shape
        N_spatial = H * W
        
        # 准备空白画布
        x_dense_flat = torch.zeros(B, N_spatial, T, C, device=x_sparse.device, dtype=x_sparse.dtype)
        
        # 扩展索引
        idx_expanded = indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, T, C)
        
        # 填充
        x_dense_flat.scatter_(1, idx_expanded, x_sparse)
        
        # 恢复形状
        x_dense = x_dense_flat.view(B, H, W, T, C).permute(0, 3, 4, 1, 2)
        
        return x_dense

class LocalityEnhancedMLP(nn.Module):
    """
    [Fallback Component] 稠密模式下使用的 MLP，带 DWConv 增强局部性。
    """
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
        # x: [B, L, C] where L = T*H*W
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
    [STMambaBlock v2.0]
    
    架构改进:
    1. Sparse Routing: 使用 LearnableSparseHandler 进行可微 Token 选择。
    2. Time Evolution: 使用 Mamba 处理时间序列 (独立处理每个空间 Token)。
    3. Global Spatial Mixing: 使用 Self-Attention 处理空间交互，替代简单的 MLP。
       这赋予了模型类似 Earthformer 的全局视野，但计算量仅限于 K 个活跃点。
    """
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
        
        # --- Mamba 配置 ---
        # Mamba 负责时间建模: Input [Batch, Time, Dim]
        # 这里我们将空间维度视为 Batch 的一部分
        mamba_cfg = dict(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.scan = Mamba(**mamba_cfg)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = RMSNorm(dim)
        
        # --- 路径分支选择 ---
        if self.target_sparse_ratio > 0:
            # 1. 可微稀疏处理器
            self.sparse_handler = LearnableSparseHandler(dim, self.target_sparse_ratio)
            
            # 2. [改进] 稀疏空间自注意力 (Sparse Self-Attention)
            # 允许 K 个活跃云团之间进行全局交互
            self.spatial_attn = nn.MultiheadAttention(
                embed_dim=dim, num_heads=4, batch_first=True, dropout=drop
            )
            self.norm_attn = nn.LayerNorm(dim)
        else:
            # Fallback: 稠密 MLP
            self.mlp = LocalityEnhancedMLP(dim, int(dim * mlp_ratio), dim, act_layer=act_layer, drop=drop)
            
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)

    def _get_curr_sparse_ratio(self, epoch):
        """计算当前的退火稀疏率"""
        if self.target_sparse_ratio <= 0: return 0.0
        if epoch < self.anneal_start: return 0.0
        if epoch >= self.anneal_end: return self.target_sparse_ratio
        progress = (epoch - self.anneal_start) / (self.anneal_end - self.anneal_start)
        return self.target_sparse_ratio * progress

    def forward(self, x_in, current_epoch=0):
        """
        Args:
            x_in: [B, T, C, H, W]
            current_epoch: 当前 Epoch，用于稀疏率退火
        """
        B, T, C, H, W = x_in.shape
        L = T * H * W
        
        curr_ratio = self._get_curr_sparse_ratio(current_epoch)
        shortcut = x_in
        
        # === 分支 A: 稀疏路径 (High Performance & Global Context) ===
        if curr_ratio > 0 and hasattr(self, 'sparse_handler'):
            # 临时覆盖 handler 的 ratio
            original_ratio = self.sparse_handler.sparse_ratio
            self.sparse_handler.sparse_ratio = curr_ratio
            
            # 1. 稀疏化 (Differentiable)
            # x_sparse: [B, K, T, C]
            x_sparse, indices, shape_info = self.sparse_handler.sparsify(x_in)
            
            # 恢复 ratio
            self.sparse_handler.sparse_ratio = original_ratio
            
            K = x_sparse.shape[1]
            
            # 2. 时间演变 (Time Mixing via Mamba)
            # 将空间点 K 视为独立的 Batch
            # Input: [B*K, T, C]
            z_time = x_sparse.reshape(B * K, T, C)
            z_time = self.norm1(z_time)
            
            if self.use_checkpoint and z_time.requires_grad:
                z_time = checkpoint(self.scan, z_time, use_reentrant=False)
            else:
                z_time = self.scan(z_time)
                
            # 3. 空间交互 (Spatial Mixing via Attention)
            # 变换维度以进行空间 Attention: [B*K, T, C] -> [B, T, K, C] -> [B*T, K, C]
            # 此时 Batch 变为 (B*T)，序列长度为 K (活跃点数量)
            z_space_in = z_time.view(B, K, T, C).permute(0, 2, 1, 3).reshape(B * T, K, C)
            
            # Global Attention: 任意两个活跃点都可以交互
            # Query=Key=Value=z_space_in
            attn_out, _ = self.spatial_attn(z_space_in, z_space_in, z_space_in)
            
            # Residual + Norm
            z_space_out = self.norm_attn(z_space_in + attn_out)
            
            # 4. 还原与反稀疏化
            # [B*T, K, C] -> [B, K, T, C]
            x_update_sparse = z_space_out.view(B, T, K, C).permute(0, 2, 1, 3) 
            
            # Densify: 未选中的点为 0
            x_update_dense = self.sparse_handler.densify(x_update_sparse, indices, shape_info)
            
            # 最终输出 (Dense Shortcut + Sparse Update)
            x_out = shortcut + self.drop_path(x_update_dense)
            
        # === 分支 B: 稠密路径 (Fallback/Warmup) ===
        else:
            # 全量计算，无稀疏化
            # [B, T, C, H, W] -> [B*H*W, T, C]
            x_norm = x_in.permute(0, 3, 4, 1, 2).reshape(B * H * W, T, C)
            x_norm = self.norm1(x_norm)
            
            if self.use_checkpoint and x_norm.requires_grad:
                z_time = checkpoint(self.scan, x_norm, use_reentrant=False)
            else:
                z_time = self.scan(x_norm)
                
            x_mamba = z_time.view(B, H, W, T, C).permute(0, 3, 4, 1, 2)
            x_mid = shortcut + self.drop_path(x_mamba)
            
            # 使用 MLP 进行局部空间混合
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
            
            # [修复] 统一物理边界条件: 'reflection'
            # 之前的 'zeros' 会导致移出边界的云团被截断，与 Loss 中的 reflection 不一致
            next_z = F.grid_sample(
                curr_z, 
                grid, 
                mode='bilinear', 
                padding_mode='reflection',  # <--- 修改点
                align_corners=False
            )
            
            curr_z = next_z
            preds.append(curr_z)
            
        return torch.stack(preds, dim=1), torch.stack(flow_maps, dim=1)