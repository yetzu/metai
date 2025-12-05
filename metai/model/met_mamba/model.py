# metai/model/met_mamba/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers.weight_init import trunc_normal_
from .modules import (
    ConvSC, 
    ResizeConv, 
    STMambaBlock, 
    TimeAlignBlock,
    AdvectiveProjection
)

# ==============================================================================
# 辅助函数 (Helper Functions)
# ==============================================================================

def sampling_generator(N, reverse=False):
    """
    生成下采样/上采样控制序列。
    例如 N=4 -> [False, True, False, True]
    用于控制 Encoder/Decoder 中哪些层执行分辨率变化。
    """
    samplings = [False, True] * (N // 2)
    if reverse: return list(reversed(samplings[:N]))
    else: return samplings[:N]

# ==============================================================================
# 编码器与解码器 (Encoder & Decoder)
# ==============================================================================

class Encoder(nn.Module):
    """
    [空间编码器]
    负责将输入的时空序列数据逐帧编码为潜在特征。
    
    Structure:
        Stem -> Stacked ConvSC Blocks (Downsampling)
    """
    def __init__(self, C_in, C_hid, N_S, spatio_kernel):
        super().__init__()
        # Stem 层：初步特征提取，扩展通道数
        self.stem = nn.Sequential(
            nn.Conv2d(C_in, C_hid, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(2, C_hid),
            nn.SiLU(inplace=True)
        )
        # 生成采样配置
        samplings = sampling_generator(N_S)
        # 堆叠 ConvSC 模块
        self.enc = nn.Sequential(
            ConvSC(C_hid, C_hid, spatio_kernel, downsampling=samplings[0]),
            *[ConvSC(C_hid, C_hid, spatio_kernel, downsampling=s) for s in samplings[1:]]
        )

    def forward(self, x):
        """
        Args:
            x: [B, T, C, H, W]
        Returns:
            latent: [B*T, C_hid, H', W'] 最终编码特征
            enc1:   [B*T, C_hid, H, W]   浅层特征 (用于 Skip Connection)
        """
        B, T, C, H, W = x.shape
        # 将时间维度合并到 Batch 维度，独立处理每一帧的空间特征
        x = x.view(B * T, C, H, W)
        x = self.stem(x) 
        
        # 获取第一层的输出用于跳跃连接
        enc1 = self.enc[0](x)
        latent = enc1
        
        # 逐层前向传播
        for i in range(1, len(self.enc)):
            latent = self.enc[i](latent)
            
        return latent, enc1


class Decoder(nn.Module):
    """
    [空间解码器]
    负责将潜在特征恢复到原始空间分辨率。
    
    Structure:
        Stacked ConvSC/ResizeConv Blocks (Upsampling) -> Readout
    """
    def __init__(self, C_hid, C_out, N_S, spatio_kernel):
        super().__init__()
        # 生成反向的采样配置
        samplings = sampling_generator(N_S, reverse=True)
        layers = []
        for i, s in enumerate(samplings):
            # 策略：最后两层使用 ResizeConv 替代转置卷积，减少棋盘格伪影
            if i < len(samplings) - 2:
                layers.append(ConvSC(C_hid, C_hid, spatio_kernel, upsampling=s))
            else:
                if s: 
                    layers.append(ResizeConv(C_hid, C_hid, spatio_kernel))
                else:
                    layers.append(ConvSC(C_hid, C_hid, spatio_kernel, upsampling=False))
                    
        self.dec = nn.Sequential(*layers)
        self.readout = nn.Conv2d(C_hid, C_out, 1)

    def forward(self, hid, skip=None):
        """
        Args:
            hid:  [B*T, C_hid, H', W'] 演变后的特征
            skip: [B*T, C_hid, H, W]   跳跃连接特征
        """
        # 前向传播直到最后一层之前
        for i in range(0, len(self.dec)-1):
            hid = self.dec[i](hid)
        
        # 融合跳跃连接特征
        if skip is not None:
            hid = hid + skip
            
        # 最后一层处理
        Y = self.dec[-1](hid)
        return self.readout(Y)

# ==============================================================================
# 演变网络 (Evolution Network)
# ==============================================================================

class EvolutionNet(nn.Module):
    """
    [演变网络] 核心模块
    
    功能：
        利用 STMambaBlock 建模时空序列的演变规律。
        
    特性：
        1. 噪声注入 (Noise Injection): 将随机噪声注入特征，增强生成多样性。
        2. 零初始化 (Zero Init): 噪声层初始化为 0，确保训练初期稳定性。
        3. 稀疏 Mamba (Sparse STMamba): 随着训练进行逐步引入稀疏计算。
    """
    def __init__(self, dim_in, dim_hid, num_layers, drop=0., drop_path=0., 
                 mamba_kwargs={}, use_checkpoint=True, max_t=64, input_resolution=(64, 64),
                 sparse_ratio=0.0,
                 anneal_start_epoch=5, anneal_end_epoch=10):
        super().__init__()
        
        # 1. 噪声投影层
        self.noise_dim = 32
        self.noise_proj = nn.Linear(self.noise_dim, dim_hid) 
        
        # [Critical Init] 初始化为 0，防止初期噪声干扰收敛
        nn.init.zeros_(self.noise_proj.weight)
        nn.init.zeros_(self.noise_proj.bias)
        
        # 2. 特征投影
        self.proj_in = nn.Linear(dim_in, dim_hid)
        self.proj_out = nn.Linear(dim_hid, dim_in)
        
        # 3. 位置编码 (Time & Space PE)
        self.pos_embed_t = nn.Parameter(torch.zeros(1, max_t, dim_hid, 1, 1)) 
        H_feat, W_feat = input_resolution
        self.pos_embed_s = nn.Parameter(torch.zeros(1, 1, dim_hid, H_feat, W_feat)) 
        self.time_prompt = nn.Parameter(torch.zeros(1, max_t, dim_hid, 1, 1))

        trunc_normal_(self.pos_embed_t, std=0.02)
        trunc_normal_(self.pos_embed_s, std=0.02)
        trunc_normal_(self.time_prompt, std=0.02)
        
        # 4. 堆叠 STMamba Block
        dpr = [x.item() for x in torch.linspace(1e-2, drop_path, num_layers)]
        self.layers = nn.ModuleList([
            STMambaBlock(
                dim_hid, 
                drop=drop, 
                drop_path=dpr[i], 
                use_checkpoint=use_checkpoint,
                sparse_ratio=sparse_ratio,
                anneal_start_epoch=anneal_start_epoch, 
                anneal_end_epoch=anneal_end_epoch,     
                **mamba_kwargs
            ) for i in range(num_layers)
        ])

    def forward(self, x, current_epoch=0):
        """
        Args:
            x: [B, T, C, H, W] 输入特征
            current_epoch: 当前训练轮次 (用于稀疏度退火)
        """
        B, T, C, H, W = x.shape
        
        # 1. 噪声生成与注入
        if self.training:
            noise = torch.randn(B, T, self.noise_dim, device=x.device)
        else:
            # 推理时默认使用零噪声 (Mean Prediction)
            noise = torch.zeros(B, T, self.noise_dim, device=x.device)
            
        # [B, T, 32] -> [B, T, C_hid] -> [B, T, C_hid, 1, 1]
        noise_emb = self.noise_proj(noise).unsqueeze(3).unsqueeze(3)
        
        # 2. 特征投影
        x = x.permute(0, 1, 3, 4, 2).contiguous() # [B, T, H, W, C]
        x = self.proj_in(x)
        x = x.permute(0, 1, 4, 2, 3).contiguous() # [B, T, C_hid, H, W]
        
        # 3. 融合噪声 (加法融合)
        x = x + noise_emb
        
        # 4. 位置编码插值与叠加
        # Time PE
        if T <= self.time_prompt.shape[1]:
            t_prompt = self.time_prompt[:, :T, ...]
            t_pos = self.pos_embed_t[:, :T, ...]
        else:
            # 线性插值适应更长序列
            t_prompt = F.interpolate(
                self.time_prompt.squeeze(-1).squeeze(-1).permute(0,2,1), 
                size=T, mode='linear'
            ).permute(0,2,1).unsqueeze(-1).unsqueeze(-1)
            t_pos = F.interpolate(
                self.pos_embed_t.squeeze(-1).squeeze(-1).permute(0,2,1), 
                size=T, mode='linear'
            ).permute(0,2,1).unsqueeze(-1).unsqueeze(-1)

        # Spatial PE
        if (H, W) != self.pos_embed_s.shape[-2:]:
            s_pos = F.interpolate(
                self.pos_embed_s, size=(H, W), mode='bilinear', align_corners=False
            )
        else:
            s_pos = self.pos_embed_s

        x = x + t_prompt + t_pos + s_pos
        
        # 5. 通过 STMamba 层
        for layer in self.layers:
            x = layer(x, current_epoch=current_epoch)
            
        # 6. 输出投影
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = self.proj_out(x)
        x = x.permute(0, 1, 4, 2, 3).contiguous()
        
        return x

# ==============================================================================
# 主模型 (Main Model Class)
# ==============================================================================

class MeteoMamba(nn.Module):
    """
    [MeteoMamba v3.0] No-GAN Version
    
    架构:
        1. Encoder: 提取空间特征
        2. EvolutionNet: 时空演变 + 随机噪声注入
        3. Advection: 显式物理平流投影 (Physics-Guided Projection)
        4. Decoder: 特征恢复与上采样
        
    特点:
        - 纯监督学习 (Rule-Based Loss 驱动)
        - 包含物理约束模块 (AdvectiveProjection)
        - 支持稀疏计算退火
    """
    def __init__(self, 
                 in_shape,      # (C, H, W)
                 in_seq_len,    # T_in
                 out_seq_len,   # T_out
                 out_channels=1,
                 hid_S=64,      # 空间隐藏层维度
                 hid_T=256,     # 时间隐藏层维度
                 N_S=4,         # 空间层数
                 N_T=8,         # 时间层数
                 spatio_kernel_enc=3, 
                 spatio_kernel_dec=3, 
                 mamba_d_state=16, 
                 mamba_d_conv=4, 
                 mamba_expand=2,
                 use_checkpoint=True,
                 mamba_sparse_ratio=0.5, 
                 anneal_start_epoch=5,
                 anneal_end_epoch=10,
                 **kwargs):
        super().__init__()
        
        C, H, W = in_shape
        T_in = in_seq_len
        self.T_out = out_seq_len
        self.out_channels = out_channels
        
        mamba_kwargs = {
            'd_state': mamba_d_state,
            'd_conv': mamba_d_conv,
            'expand': mamba_expand
        }
        
        # 1. 初始化编码器
        self.enc = Encoder(C, hid_S, N_S, spatio_kernel_enc)
        
        # 计算演变阶段的空间分辨率
        ds_factor = 2 ** (N_S // 2) 
        evo_res = (H // ds_factor, W // ds_factor)

        # 2. 初始化演变网络
        self.evolution = EvolutionNet(
            hid_S, hid_T, N_T, 
            drop=0.0, drop_path=0.1, 
            mamba_kwargs=mamba_kwargs,
            use_checkpoint=use_checkpoint,
            input_resolution=evo_res,
            sparse_ratio=mamba_sparse_ratio,
            anneal_start_epoch=anneal_start_epoch, 
            anneal_end_epoch=anneal_end_epoch      
        )
        
        # 3. 显式平流时间投影层 (物理约束)
        self.latent_time_proj = AdvectiveProjection(
            dim=hid_S, 
            t_in=T_in, 
            t_out=self.T_out
        )
        
        # 4. 初始化解码器
        self.dec = Decoder(hid_S, self.out_channels, N_S, spatio_kernel_dec)
        
        # 5. 跳跃连接投影层 (用于连接 Encoder 第一层与 Decoder 倒数第二层)
        self.skip_proj = TimeAlignBlock(T_in, self.T_out, hid_S)

    def forward(self, x_raw, current_epoch=0):
        """
        Args:
            x_raw: [B, T_in, C_in, H, W] 原始输入序列
            current_epoch: 当前 Epoch
            
        Returns:
            Y: [B, T_out, C_out, H, W] 预测结果
            flows: [B, T_out, 2, H, W] 潜在流场
        """
        B, T_in, C_in, H, W = x_raw.shape
        
        # 1. 编码阶段 (Encoder)
        # embed: [B*T_in, C_hid, H', W'], skip: [B*T_in, C_hid, H, W]
        embed, skip = self.enc(x_raw) 
        _, C_hid, H_, W_ = embed.shape 
        
        # 2. 演变阶段 (Evolution)
        z = embed.view(B, T_in, C_hid, H_, W_)
        z = self.evolution(z, current_epoch=current_epoch)
        
        # 3. 时间投影阶段 (Time Projection via Advection)
        # z: [B, T_out, C_hid, H', W'], flows: [B, T_out, 2, H', W']
        z, flows = self.latent_time_proj(z) 
        
        # 重塑为 Decoder 需要的格式
        z = z.view(B * self.T_out, C_hid, H_, W_)
        
        # 4. 处理跳跃连接 (Skip Connection)
        # 调整 skip 特征的时间维度以匹配输出序列长度
        skip = skip.view(B, T_in, C_hid, H, W)
        skip_out = self.skip_proj(skip) 
        skip_out = skip_out.view(B * self.T_out, C_hid, H, W)
        
        # 5. 解码阶段 (Decoder)
        Y_diff = self.dec(z, skip_out)
        Y_diff = Y_diff.reshape(B, self.T_out, self.out_channels, H, W)
        
        # 6. 残差连接 (Residual Connection)
        # 预测的是相对于最后一帧的变化量
        last_frame = x_raw[:, -1:, :self.out_channels, :, :].detach() 
        Y = last_frame + Y_diff
        
        return Y, flows