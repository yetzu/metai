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
# 判别器模块 (Discriminator for GAN Training)
# ==============================================================================

class TemporalDiscriminator(nn.Module):
    """
    [新增组件] 时空判别器
    
    功能：
    基于 3D 卷积的判别网络，用于区分模型生成的雷达回波序列与真实观测序列。
    它关注时空纹理的真实性，迫使生成器恢复短时强降水的锐利边界和极值强度。
    
    架构特点：
    1. 使用 Spectral Normalization (谱归一化) 稳定 GAN 的训练动态。
    2. 输出为 PatchGAN 风格的 Logits Map，而非单一标量，关注局部真实性。
    """
    def __init__(self, in_channels=1, base_dim=64):
        super().__init__()
        
        # 定义带谱归一化的 3D 卷积
        def sn_conv3d(in_c, out_c, k, s, p):
            return nn.utils.spectral_norm(nn.Conv3d(in_c, out_c, k, s, p))

        self.blocks = nn.Sequential(
            # Input: [B, C, T, H, W]
            # Layer 1: 下采样 -> [B, 64, T/2, H/2, W/2]
            sn_conv3d(in_channels, base_dim, 4, 2, 1), 
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 2: 下采样 -> [B, 128, T/4, H/4, W/4]
            sn_conv3d(base_dim, base_dim*2, 4, 2, 1), 
            nn.InstanceNorm3d(base_dim*2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 3: 下采样 -> [B, 256, T/8, H/8, W/8]
            sn_conv3d(base_dim*2, base_dim*4, 4, 2, 1), 
            nn.InstanceNorm3d(base_dim*4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 4: 特征提取
            sn_conv3d(base_dim*4, base_dim*8, 4, 2, 1),
            nn.InstanceNorm3d(base_dim*8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # Output Head: 映射到判别分数
        self.out_head = nn.Conv3d(base_dim*8, 1, kernel_size=1) 

    def forward(self, x):
        """
        Args:
            x: [B, T, C, H, W] (生成器输出或真实标签)
        Returns:
            logits: [B, 1, T', H', W']
        """
        # 调整维度适配 Conv3d: [B, T, C, H, W] -> [B, C, T, H, W]
        x = x.permute(0, 2, 1, 3, 4)
        feat = self.blocks(x)
        logits = self.out_head(feat)
        return logits

# ==============================================================================
# 编码器与解码器 (Encoder & Decoder)
# ==============================================================================

class Encoder(nn.Module):
    """
    空间编码器 (Spatial Encoder)
    负责将输入的时空序列数据 [B, T, C, H, W] 逐帧编码为潜在特征。
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
        # 堆叠 ConvSC 模块，根据配置进行下采样
        self.enc = nn.Sequential(
            ConvSC(C_hid, C_hid, spatio_kernel, downsampling=samplings[0]),
            *[ConvSC(C_hid, C_hid, spatio_kernel, downsampling=s) for s in samplings[1:]]
        )

    def forward(self, x):
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        # 将时间维度合并到 Batch 维度，独立处理每一帧的空间特征
        x = x.view(B * T, C, H, W)
        x = self.stem(x) 
        
        # 获取第一层的输出用于跳跃连接 (Skip Connection)
        enc1 = self.enc[0](x)
        latent = enc1
        
        # 逐层前向传播
        for i in range(1, len(self.enc)):
            latent = self.enc[i](latent)
            
        # 返回最终的潜在特征和浅层特征(用于Skip connection)
        return latent, enc1


class Decoder(nn.Module):
    """
    空间解码器 (Spatial Decoder)
    负责将潜在特征恢复到原始空间分辨率。
    """
    def __init__(self, C_hid, C_out, N_S, spatio_kernel):
        super().__init__()
        # 生成反向的采样配置
        samplings = sampling_generator(N_S, reverse=True)
        layers = []
        for i, s in enumerate(samplings):
            # 策略：最后两层使用 ResizeConv 替代转置卷积/PixelShuffle，以减少棋盘格伪影 (checkerboard artifacts)
            if i < len(samplings) - 2:
                layers.append(ConvSC(C_hid, C_hid, spatio_kernel, upsampling=s))
            else:
                if s: 
                    # 如果需要上采样
                    layers.append(ResizeConv(C_hid, C_hid, spatio_kernel))
                else:
                    layers.append(ConvSC(C_hid, C_hid, spatio_kernel, upsampling=False))
                    
        self.dec = nn.Sequential(*layers)
        # 最后的输出投影层，将通道数调整回 C_out
        self.readout = nn.Conv2d(C_hid, C_out, 1)

    def forward(self, hid, skip=None):
        # 前向传播直到最后一层之前
        for i in range(0, len(self.dec)-1):
            hid = self.dec[i](hid)
        
        # 融合跳跃连接特征 (Skip Connection Fusion)
        if skip is not None:
            hid = hid + skip
            
        # 最后一层处理
        Y = self.dec[-1](hid)
        # 输出投影
        return self.readout(Y)

# ==============================================================================
# 演变网络 (Evolution Network)
# ==============================================================================

class EvolutionNet(nn.Module):
    """
    演变网络 (Evolution Network) - v2.0
    
    核心模块，利用 STMambaBlock 建模时空序列的演变规律。
    
    [改进特性] 噪声注入 (Noise Injection):
    在输入特征中叠加随机噪声，使确定性的 Mamba 网络转变为概率生成模型。
    这允许模型从"平均分布"中采样出具体的"纹理实例"，解决模糊问题。
    """
    def __init__(self, dim_in, dim_hid, num_layers, drop=0., drop_path=0., 
                 mamba_kwargs={}, use_checkpoint=True, max_t=64, input_resolution=(64, 64),
                 sparse_ratio=0.0,
                 anneal_start_epoch=5, anneal_end_epoch=10):
        super().__init__()
        
        # [新增] 噪声投影层：将低维随机噪声映射到特征空间
        # 假设噪声源维度为 32 (可配置)
        self.noise_dim = 32
        self.noise_proj = nn.Linear(self.noise_dim, dim_hid) 
        
        # 特征投影层
        self.proj_in = nn.Linear(dim_in, dim_hid)
        self.proj_out = nn.Linear(dim_hid, dim_in)
        
        # 3D 绝对位置编码 (Temporal PE)
        self.pos_embed_t = nn.Parameter(torch.zeros(1, max_t, dim_hid, 1, 1)) 
        
        # 动态空间位置编码初始化 (Spatial PE)
        H_feat, W_feat = input_resolution
        self.pos_embed_s = nn.Parameter(torch.zeros(1, 1, dim_hid, H_feat, W_feat)) 
        
        # 时间提示 (Time Prompt)
        self.time_prompt = nn.Parameter(torch.zeros(1, max_t, dim_hid, 1, 1))

        # 权重初始化
        trunc_normal_(self.pos_embed_t, std=0.02)
        trunc_normal_(self.pos_embed_s, std=0.02)
        trunc_normal_(self.time_prompt, std=0.02)
        
        # Drop path设置 (随机深度)
        dpr = [x.item() for x in torch.linspace(1e-2, drop_path, num_layers)]
        
        # 堆叠 STMambaBlock (已包含 Global Attention 和 Sparse Routing)
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
            current_epoch: 当前训练轮次
        """
        B, T, C, H, W = x.shape
        
        # 1. 噪声生成与注入 (Stochasticity)
        # 策略：训练时注入随机噪声；推理时为了确定性基准可注入零噪声，
        #      或者为了生成多样性结果注入随机噪声。
        if self.training:
            noise = torch.randn(B, T, self.noise_dim, device=x.device)
        else:
            # 默认推理使用零噪声 (Mean Prediction)
            # 若需多样性生成，此处可改为 torch.randn
            noise = torch.zeros(B, T, self.noise_dim, device=x.device)
            
        # 映射噪声: [B, T, 32] -> [B, T, C_hid] -> [B, T, C_hid, 1, 1]
        noise_emb = self.noise_proj(noise).unsqueeze(3).unsqueeze(3)
        
        # 2. 特征投影
        # 调整维度以适应 Linear 层: [B, T, H, W, C]
        x = x.permute(0, 1, 3, 4, 2).contiguous() 
        x = self.proj_in(x)
        x = x.permute(0, 1, 4, 2, 3).contiguous() # 回到 [B, T, C_hid, H, W]
        
        # 3. 融合噪声 (加法融合)
        x = x + noise_emb
        
        # --- 位置编码插值逻辑 ---
        # 4. 处理时间提示 (Time Prompt) 和 时间位置编码 (Time PE)
        if T <= self.time_prompt.shape[1]:
            t_prompt = self.time_prompt[:, :T, ...]
            t_pos = self.pos_embed_t[:, :T, ...]
        else:
            t_prompt = F.interpolate(
                self.time_prompt.squeeze(-1).squeeze(-1).permute(0,2,1), 
                size=T, mode='linear'
            ).permute(0,2,1).unsqueeze(-1).unsqueeze(-1)
            t_pos = F.interpolate(
                self.pos_embed_t.squeeze(-1).squeeze(-1).permute(0,2,1), 
                size=T, mode='linear'
            ).permute(0,2,1).unsqueeze(-1).unsqueeze(-1)

        # 5. 处理空间位置编码 (Spatial PE)
        if (H, W) != self.pos_embed_s.shape[-2:]:
            s_pos = F.interpolate(
                self.pos_embed_s, size=(H, W), mode='bilinear', align_corners=False
            )
        else:
            s_pos = self.pos_embed_s

        # 6. 将位置编码和提示叠加到特征上
        x = x + t_prompt + t_pos + s_pos
        
        # 7. 通过 STMamba 层进行演变
        for layer in self.layers:
            # 将 current_epoch 传递给 Block 用于稀疏率退火
            x = layer(x, current_epoch=current_epoch)
            
        # 8. 输出投影
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = self.proj_out(x)
        x = x.permute(0, 1, 4, 2, 3).contiguous()
        return x

# ==============================================================================
# 主模型 (Main Model Class)
# ==============================================================================

class MeteoMamba(nn.Module):
    """
    MeteoMamba 主模型类 - v2.0 (Generative & Sparse)
    
    整体架构：
    Encoder -> EvolutionNet (with Noise Injection & Mamba) -> Advective Time Projection -> Decoder
    
    集成特性：
    1. 物理约束 (Physical Alignment): 显式平流层 (AdvectiveProjection)。
    2. 生成能力 (Generative): 内置 Discriminator 和 EvolutionNet 噪声注入。
    3. 全局视野 (Global Context): 通过 STMambaBlock 中的 Attention 实现。
    """
    def __init__(self, 
                 in_shape,      # 输入形状 (C, H, W)
                 in_seq_len,    # 输入序列长度 T_in
                 out_seq_len,   # 预测序列长度 T_out
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

        # 2. 初始化演变网络 (支持噪声注入)
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
        
        # 5. 跳跃连接投影层
        self.skip_proj = TimeAlignBlock(T_in, self.T_out, hid_S)

        # 6. [新增] 判别器 (Discriminator)
        # 注意：Discriminator 是模型的一部分，但在训练时需要使用不同的优化器更新
        self.discriminator = TemporalDiscriminator(in_channels=self.out_channels)

    def forward(self, x_raw, current_epoch=0):
        """
        Args:
            x_raw: [B, T_in, C_in, H, W] 原始输入序列
            current_epoch: 当前 Epoch (用于稀疏率调度)
            
        Returns:
            Y: [B, T_out, C_out, H, W] 预测结果
            flows: [B, T_out, 2, H, W] 潜在流场 (用于物理约束 Loss)
        """
        # x_raw 输入形状: [B, T_in, C_in, H, W]
        B, T_in, C_in, H, W = x_raw.shape
        
        # 1. 编码阶段 (Encoder)
        embed, skip = self.enc(x_raw) 
        _, C_hid, H_, W_ = embed.shape 
        
        # 2. 演变阶段 (Evolution with Noise)
        z = embed.view(B, T_in, C_hid, H_, W_)
        z = self.evolution(z, current_epoch=current_epoch)
        
        # 3. 时间投影阶段 (Time Projection via Advection)
        # 返回演变后的特征 z 和 预测的流场 flows
        z, flows = self.latent_time_proj(z) 
        
        # 重塑为 Decoder 需要的格式: [B*T_out, C_hid, H_, W_]
        z = z.view(B * self.T_out, C_hid, H_, W_)
        
        # 4. 处理跳跃连接 (Skip Connection)
        skip = skip.view(B, T_in, C_hid, H, W)
        skip_out = self.skip_proj(skip) 
        skip_out = skip_out.view(B * self.T_out, C_hid, H, W)
        
        # 5. 解码阶段 (Decoder)
        Y_diff = self.dec(z, skip_out)
        Y_diff = Y_diff.reshape(B, self.T_out, self.out_channels, H, W)
        
        # 6. 残差连接 (Residual Connection)
        # 基于最后一帧进行增量预测
        last_frame = x_raw[:, -1:, :self.out_channels, :, :].detach() 
        Y = last_frame + Y_diff
        
        return Y, flows