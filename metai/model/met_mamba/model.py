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
    AdvectiveProjection  # [新增] 引入显式平流投影层
)

def sampling_generator(N, reverse=False):
    """
    生成下采样/上采样控制序列。
    例如 N=4 -> [False, True, False, True]
    用于控制 Encoder/Decoder 中哪些层执行分辨率变化。
    """
    samplings = [False, True] * (N // 2)
    if reverse: return list(reversed(samplings[:N]))
    else: return samplings[:N]

class Encoder(nn.Module):
    """
    空间编码器 (Spatial Encoder)
    负责将输入的时空序列数据 [B, T, C, H, W] 编码为潜在特征。
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

class EvolutionNet(nn.Module):
    """
    演变网络 (Evolution Network)
    核心模块，利用 Mamba (SSM) 建模时空序列的演变规律。
    """
    def __init__(self, dim_in, dim_hid, num_layers, drop=0., drop_path=0., 
                 mamba_kwargs={}, use_checkpoint=True, max_t=64, input_resolution=(64, 64)):
        super().__init__()
        # 特征投影层
        self.proj_in = nn.Linear(dim_in, dim_hid)
        self.proj_out = nn.Linear(dim_hid, dim_in)
        
        # 3D 绝对位置编码 (Temporal PE)
        self.pos_embed_t = nn.Parameter(torch.zeros(1, max_t, dim_hid, 1, 1)) 
        
        # 动态空间位置编码初始化 (Spatial PE)
        H_feat, W_feat = input_resolution
        self.pos_embed_s = nn.Parameter(torch.zeros(1, 1, dim_hid, H_feat, W_feat)) 
        
        # 时间提示 (Time Prompt): 针对稀疏雷达数据的优化，增强模型对时间节点的感知
        self.time_prompt = nn.Parameter(torch.zeros(1, max_t, dim_hid, 1, 1))

        # 权重初始化
        trunc_normal_(self.pos_embed_t, std=0.02)
        trunc_normal_(self.pos_embed_s, std=0.02)
        trunc_normal_(self.time_prompt, std=0.02)
        
        # Drop path设置 (随机深度)
        dpr = [x.item() for x in torch.linspace(1e-2, drop_path, num_layers)]
        # 堆叠 STMambaBlock
        self.layers = nn.ModuleList([
            STMambaBlock(
                dim_hid, 
                drop=drop, 
                drop_path=dpr[i], 
                use_checkpoint=use_checkpoint, 
                **mamba_kwargs
            ) for i in range(num_layers)
        ])

    def forward(self, x):
        # 输入 x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        
        # 调整维度以适应 Linear 层: [B, T, H, W, C]
        x = x.permute(0, 1, 3, 4, 2).contiguous() 
        x = self.proj_in(x)
        x = x.permute(0, 1, 4, 2, 3).contiguous() # 回到 [B, T, C_hid, H, W]
        
        # --- 位置编码插值逻辑 ---
        # 1. 处理时间提示 (Time Prompt) 和 时间位置编码 (Time PE)
        # 如果当前序列长度 T 小于预设 max_t，直接切片；否则进行线性插值
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

        # 2. 处理空间位置编码 (Spatial PE)
        # 如果特征图尺寸与预设不符，进行双线性插值
        if (H, W) != self.pos_embed_s.shape[-2:]:
            s_pos = F.interpolate(
                self.pos_embed_s, size=(H, W), mode='bilinear', align_corners=False
            )
        else:
            s_pos = self.pos_embed_s

        # 3. 将位置编码和提示叠加到特征上
        x = x + t_prompt + t_pos + s_pos
        
        # 通过 Mamba 层进行演变
        for layer in self.layers:
            x = layer(x)
            
        # 输出投影
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = self.proj_out(x)
        x = x.permute(0, 1, 4, 2, 3).contiguous()
        return x

class MeteoMamba(nn.Module):
    """
    MeteoMamba 主模型类
    整体架构：Encoder -> EvolutionNet (Latent Space) -> Advective Time Projection -> Decoder
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
        
        # 计算演变阶段的空间分辨率 (假设 N_S=4 意味着下采样 2 次，即 4 倍)
        ds_factor = 2 ** (N_S // 2) 
        evo_res = (H // ds_factor, W // ds_factor)

        # 2. 初始化演变网络
        self.evolution = EvolutionNet(
            hid_S, hid_T, N_T, 
            drop=0.0, drop_path=0.1, 
            mamba_kwargs=mamba_kwargs,
            use_checkpoint=use_checkpoint,
            input_resolution=evo_res
        )
        
        # 3. [修改] 显式平流时间投影层 (Advective Time Projection)
        # 替代原有的 Conv1d 投影。
        # 这里的 dim 对应 Encoder 的输出通道数 hid_S
        self.latent_time_proj = AdvectiveProjection(
            dim=hid_S, 
            t_in=T_in, 
            t_out=self.T_out
        )
        
        # 4. 初始化解码器
        self.dec = Decoder(hid_S, self.out_channels, N_S, spatio_kernel_dec)
        
        # 5. 跳跃连接投影层 (用于对齐 Encoder 特征的时间维度)
        self.skip_proj = TimeAlignBlock(T_in, self.T_out, hid_S)
        
        # 注：AdvectiveProjection 内部已处理初始化，无需额外的 _init_time_proj

    def forward(self, x_raw):
        # x_raw 输入形状: [B, T_in, C_in, H, W]
        B, T_in, C_in, H, W = x_raw.shape
        
        # 1. 编码阶段 (Encoder)
        # 将时空数据编码为潜在特征和浅层特征 (用于Skip)
        # embed: [B*T_in, C_hid, H_, W_]
        embed, skip = self.enc(x_raw) 
        _, C_hid, H_, W_ = embed.shape 
        
        # 2. 演变阶段 (Evolution)
        # 恢复时间维度，通过 Mamba 建模时空演变
        # z: [B, T_in, C_hid, H_, W_]
        z = embed.view(B, T_in, C_hid, H_, W_)
        z = self.evolution(z)
        
        # 3. [修改] 时间投影阶段 (Time Projection)
        # 使用 AdvectiveProjection 进行流引导的特征变换
        # 输入: [B, T_in, C_hid, H_, W_]
        # 输出: [B, T_out, C_hid, H_, W_]
        z = self.latent_time_proj(z) 
        
        # 重塑为 Decoder 需要的格式: [B*T_out, C_hid, H_, W_]
        z = z.view(B * self.T_out, C_hid, H_, W_)
        
        # 4. 处理跳跃连接 (Skip Connection)
        # 将 Encoder 的浅层特征的时间维度也从 T_in 投影到 T_out
        skip = skip.view(B, T_in, C_hid, H, W)
        skip_out = self.skip_proj(skip) 
        skip_out = skip_out.view(B * self.T_out, C_hid, H, W)
        
        # 5. 解码阶段 (Decoder)
        # 结合演变后的特征和跳跃连接，恢复空间分辨率
        # Y_diff: [B*T_out, C_out, H, W] -> [B, T_out, C_out, H, W]
        Y_diff = self.dec(z, skip_out)
        Y_diff = Y_diff.reshape(B, self.T_out, self.out_channels, H, W)
        
        # 6. 残差连接 (Residual Connection)
        # 预测模式：当前帧 = 最后一帧观测值 + 预测的变化量
        last_frame = x_raw[:, -1:, :self.out_channels, :, :].detach() 
        Y = last_frame + Y_diff
        
        return Y