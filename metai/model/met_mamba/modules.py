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
# 稀疏计算组件 (Sparse Computation Components)
# ==========================================

class SparseTokenHandler(nn.Module):
    """
    [稀疏 Token 处理器 - 改进版]
    实现真正的计算稀疏化：对 Top-K 活跃 Token 和部分随机 Token 进行计算。
    """
    def __init__(self, sparse_ratio=0.5, random_ratio=0.1):
        super().__init__()
        self.sparse_ratio = sparse_ratio
        self.random_ratio = random_ratio # 预留给随机采样的比例 (相对于 keep_len)

    def sparsify(self, x):
        """
        Dense [B, L, C] -> Sparse [B, K, C]
        返回: x_sparse, indices, shape_info
        """
        B, L, C = x.shape
        # 计算 Token 重要性 (L2 Norm)
        x_norm = x.norm(dim=-1)
        
        # 确定总保留数量 K
        # 至少保留 1 个 token 避免崩溃
        keep_len = max(1, int(L * (1.0 - self.sparse_ratio)))
        
        # [修改] 混合采样策略: Top-K (强信号) + Random (弱信号探查)
        if self.random_ratio > 0 and keep_len > 1:
            num_random = int(keep_len * self.random_ratio)
            num_top = keep_len - num_random
            num_top = max(1, num_top) # 确保至少有一个 Top
            num_random = max(0, keep_len - num_top)
        else:
            num_top = keep_len
            num_random = 0
            
        # 1. 获取 Top-K 索引 (强信号)
        _, top_indices = torch.topk(x_norm, k=num_top, dim=1)
        
        # 2. 获取随机索引 (弱信号)
        if num_random > 0:
            # 创建一个 mask，排除掉已经被选为 top 的 indices
            mask = torch.ones(B, L, device=x.device, dtype=torch.bool)
            # scatter 将 top_indices 位置置为 False
            mask.scatter_(1, top_indices, False)
            
            # 从剩余位置中采样
            # 由于 torch.multinomial 需要 float 概率，我们将 mask 转为 float
            # 注意：这里对每个样本分别采样
            rand_indices_list = []
            for b in range(B):
                # 获取当前样本剩余的索引
                remaining_indices = torch.nonzero(mask[b], as_tuple=False).squeeze(-1)
                if len(remaining_indices) >= num_random:
                    # 随机选择
                    perm = torch.randperm(len(remaining_indices), device=x.device)[:num_random]
                    chosen = remaining_indices[perm]
                else:
                    # 如果剩余不够 (极少见)，就全选或补全
                    chosen = remaining_indices
                rand_indices_list.append(chosen)
            
            rand_indices = torch.stack(rand_indices_list, dim=0)
            
            # 合并索引
            indices = torch.cat([top_indices, rand_indices], dim=1)
        else:
            indices = top_indices
        
        # 排序索引，尽量保持原有的相对顺序 (这对 Mamba 的时序性有帮助)
        indices, _ = torch.sort(indices, dim=1)
        
        # Gather 提取 Active Tokens
        batch_indices = torch.arange(B, device=x.device).unsqueeze(1).expand(-1, indices.shape[1])
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

class LocalityEnhancedMLP(nn.Module):
    """
    [标准局部增强 MLP]
    Linear -> DW-Conv -> Linear
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
    """
    [物理感知稀疏 MLP] (Sandwich Architecture)
    通过"稀疏-稠密-稀疏"的三明治结构，在保留局部性的同时最大化计算效率。
    1. Sparse Linear (升维): 仅计算有效 Token
    2. Dense DW-Conv (局部性): 临时恢复局部网格计算梯度
    3. Sparse Linear (降维): 仅计算有效 Token
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        # 1. Sparse Linear
        self.fc1 = nn.Linear(in_features, hidden_features)
        
        # 2. Dense DW-Conv (Groups=C means depth-wise)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)
        
        self.act = act_layer()
        
        # 3. Sparse Linear
        self.fc2 = nn.Linear(hidden_features, out_features)
        
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, x_sparse, indices, B, L, H, W):
        """
        x_sparse: [B, K, C] (Active Tokens)
        indices:  [B, K]    (Token Indices)
        """
        K = x_sparse.shape[1]
        
        # --- Step 1: Sparse Linear Up ---
        x = self.fc1(x_sparse) # [B, K, C_hid]
        
        # --- Step 2: Densify for Locality ---
        x_dense = torch.zeros(B, L, x.shape[-1], device=x.device, dtype=x.dtype)
        batch_indices = torch.arange(B, device=x.device).unsqueeze(1).expand(-1, K)
        x_dense[batch_indices, indices] = x
        
        # Reshape to 2D: [B*T, C_hid, H, W]
        T = L // (H * W)
        x_spatial = x_dense.view(B * T, H, W, -1).permute(0, 3, 1, 2)
        
        # --- Step 3: Dense DW-Conv ---
        x_spatial = self.dwconv(x_spatial)
        
        # --- Step 4: Re-Sparsify (Gather) ---
        x_flat = x_spatial.permute(0, 2, 3, 1).reshape(B, L, -1)
        x_sparse_new = x_flat[batch_indices, indices] # Gather back
        
        x_sparse_new = self.act(x_sparse_new)
        x_sparse_new = self.drop(x_sparse_new)
        
        # --- Step 5: Sparse Linear Down ---
        x = self.fc2(x_sparse_new)
        x = self.drop(x)
        
        return x

# ==========================================
# 增强型 Mamba 模块 (Enhanced Mamba Modules)
# ==========================================

class STMambaBlock(nn.Module):
    """
    [稀疏感知时空 Mamba 模块] (Final Version)
    集成 "Sandwich" 架构，同时实现 Mamba 和 MLP 的稀疏化计算。
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
        
        # [动态选择 MLP 类型]
        if self.sparse_ratio > 0:
            # 稀疏模式：使用 Sandwich MLP
            self.mlp = SparseLocalityEnhancedMLP(dim, mlp_hidden_dim, dim, act_layer=act_layer, drop=drop)
            # 使用默认 random_ratio=0.1 进行混合采样
            self.sparse_handler = SparseTokenHandler(sparse_ratio)
        else:
            # 稠密模式：使用标准 MLP
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
        
        # 展平: [B, L, C]
        x = x_in.permute(0, 1, 3, 4, 2).contiguous().reshape(B, L, C)
        
        shortcut = x
        x_norm = self.norm1(x)

        # === 稀疏计算路径 (Sandwich Path) ===
        if self.sparse_ratio > 0:
            # 1. 剪枝 (Sparsify based on Norm1)
            # x_sparse: [B, K, C] (Normalized values)
            x_sparse, indices, _ = self.sparse_handler.sparsify(x_norm)
            
            # 2. 稀疏 Mamba 扫描
            def sparse_scan(z):
                return self.scan(z) + self.scan(z.flip([1])).flip([1])

            if self.use_checkpoint and x_sparse.requires_grad:
                mamba_out_sparse = checkpoint(sparse_scan, x_sparse, use_reentrant=False)
            else:
                mamba_out_sparse = sparse_scan(x_sparse)
            
            # 3. 稀疏 MLP (Reuse Indices)
            # 构造 MLP 输入: Norm2(x + Mamba(x))
            # 我们需要获取 x 在 active indices 处的值
            batch_indices = torch.arange(B, device=x.device).unsqueeze(1).expand(-1, indices.shape[1])
            x_val_at_indices = x[batch_indices, indices]
            
            # 中间残差状态 (Sparse Domain)
            mid_sparse = x_val_at_indices + self.drop_path(mamba_out_sparse)
            
            # MLP 输入归一化
            mlp_in_sparse = self.norm2(mid_sparse)
            
            # 执行稀疏 Sandwich MLP
            mlp_out_sparse = self.mlp(mlp_in_sparse, indices, B, L, H, W)
            
            # 总的稀疏更新量
            total_sparse_update = self.drop_path(mamba_out_sparse) + self.drop_path(mlp_out_sparse)
            
            # 4. 统一还原 (Densify once)
            update_dense = self.sparse_handler.densify(total_sparse_update, indices, (B, L, C))
            
            # 最终加回原始 x
            x = shortcut + update_dense
            
        else:
            # === 稠密计算路径 (Legacy Dense Path) ===
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

            x = shortcut + self.drop_path(out)
            x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        
        return x.reshape(B, T, H, W, C).permute(0, 1, 4, 2, 3).contiguous()

# ==========================================
# 增强型时间投影组件 (Advective Projection - Final)
# ==========================================

class FlowPredictorGRU(nn.Module):
    """
    [动态流场预测器 - GRU版]
    解决"线性假设"缺陷。
    利用 ConvGRU 单元，在每一步根据当前特征状态动态更新流场。
    """
    def __init__(self, in_channels, hidden_dim):
        super().__init__()
        # ConvGRU Gates: Update(z), Reset(r)
        self.conv_z = nn.Conv2d(in_channels + hidden_dim, hidden_dim, 3, 1, 1)
        self.conv_r = nn.Conv2d(in_channels + hidden_dim, hidden_dim, 3, 1, 1)
        # Candidate hidden state
        self.conv_h = nn.Conv2d(in_channels + hidden_dim, hidden_dim, 3, 1, 1)
        
        # 流场输出头: 预测流场的"增量" (delta flow)
        self.flow_head = nn.Conv2d(hidden_dim, 2, 3, 1, 1)
        # 初始化为0，保证初始状态平稳
        nn.init.constant_(self.flow_head.weight, 0)
        nn.init.constant_(self.flow_head.bias, 0)

    def forward(self, x, h):
        """
        x: 当前时刻特征 [B, C, H, W]
        h: 上一时刻的隐状态 [B, C_hid, H, W]
        """
        combined = torch.cat([x, h], dim=1)
        z = torch.sigmoid(self.conv_z(combined))
        r = torch.sigmoid(self.conv_r(combined))
        
        combined_new = torch.cat([x, r * h], dim=1)
        h_tilde = torch.tanh(self.conv_h(combined_new))
        
        h_next = (1 - z) * h + z * h_tilde
        delta_flow = self.flow_head(h_next)