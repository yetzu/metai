# metai/model/met_mamba/model.py

import torch
import torch.nn as nn
from timm.layers.weight_init import trunc_normal_

from .modules import (
    ConvSC, 
    ResizeConv, 
    STMambaBlock, 
    TimeAlignBlock, 
    AdvectiveProjection, 
    PosteriorNet
)

# ==============================================================================
# 编码器与解码器
# ==============================================================================

class Encoder(nn.Module):
    """
    空间编码器：将高分辨率观测数据压缩为低维隐特征。
    结构：Stem -> 多层下采样 ConvSC
    """
    def __init__(self, C_in, C_hid, N_S, spatio_kernel):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(C_in, C_hid, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(2, C_hid),
            nn.SiLU(inplace=True)
        )
        
        # 生成下采样控制序列 (True/False)
        samplings = [False, True] * (N_S // 2)
        
        layers = []
        # 第一层可能涉及下采样
        layers.append(ConvSC(C_hid, C_hid, spatio_kernel, downsampling=samplings[0]))
        # 后续层
        for s in samplings[1:]:
            layers.append(ConvSC(C_hid, C_hid, spatio_kernel, downsampling=s))
            
        self.enc = nn.Sequential(*layers)

    def forward(self, x):
        # Input: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        
        x = self.stem(x) 
        enc1 = self.enc[0](x) # 保留浅层特征用于 Skip Connection
        
        latent = enc1
        for i in range(1, len(self.enc)):
            latent = self.enc[i](latent)
            
        return latent, enc1 # latent: [B*T, C_hid, H', W']

class Decoder(nn.Module):
    """
    空间解码器：将演变后的隐特征恢复为预测图。
    结构：多层上采样 ConvSC -> Readout
    """
    def __init__(self, C_hid, C_out, N_S, spatio_kernel):
        super().__init__()
        # 生成上采样控制序列 (反向)
        samplings = list(reversed([False, True] * (N_S // 2)))
        
        layers = []
        for i, s in enumerate(samplings):
            if i < len(samplings) - 2:
                layers.append(ConvSC(C_hid, C_hid, spatio_kernel, upsampling=s))
            else:
                # 倒数第二层处理
                if s: layers.append(ResizeConv(C_hid, C_hid, spatio_kernel))
                else: layers.append(ConvSC(C_hid, C_hid, spatio_kernel, upsampling=False))
                    
        self.dec = nn.Sequential(*layers)
        self.readout = nn.Conv2d(C_hid, C_out, 1)

    def forward(self, hid, skip=None):
        # 解码主体
        for i in range(0, len(self.dec)-1):
            hid = self.dec[i](hid)
            
        # 跳跃连接融合 (最后几层前)
        if skip is not None:
            hid = hid + skip
            
        # 最后一层与输出投影
        Y = self.dec[-1](hid)
        return self.readout(Y)

# ==============================================================================
# 演变网络 (EvolutionNet)
# ==============================================================================

class EvolutionNet(nn.Module):
    """
    时空演变网络核心。
    功能：结合 CVAE 隐变量 z，利用 STMamba 推演未来特征。
    """
    def __init__(self, dim, num_layers, out_seq_len, noise_dim=32, drop_path=0.1, **kwargs):
        super().__init__()
        self.noise_dim = noise_dim
        
        # 1. 噪声投影层：将隐变量 z 映射到特征维度
        self.z_proj = nn.Linear(noise_dim, dim)
        
        # 2. 未来查询向量 (Learned Queries)：承载未来预测的容器
        self.future_tokens = nn.Parameter(torch.zeros(1, out_seq_len, dim, 1, 1))
        trunc_normal_(self.future_tokens, std=0.02)
        
        # 3. 时间位置编码
        self.pos_embed_t = nn.Parameter(torch.zeros(1, 64, dim, 1, 1)) 
        trunc_normal_(self.pos_embed_t, std=0.02)
        
        # 4. 堆叠 STMamba Block
        dpr = [x.item() for x in torch.linspace(1e-2, drop_path, num_layers)]
        self.layers = nn.ModuleList([
            STMambaBlock(dim, drop_path=dpr[i], **kwargs) for i in range(num_layers)
        ])

    def forward(self, x_hist, z_sample):
        """
        Args:
            x_hist: [B, T_in, C, H, W] 历史特征
            z_sample: [B, noise_dim] 采样的隐变量
        Returns:
            x_hist_out, x_future_out
        """
        B, T_in, C, H, W = x_hist.shape
        T_out = self.future_tokens.shape[1]
        
        # 1. 注入噪声 (Conditioning)
        # z 广播到全图，代表全局环境不确定性
        z_emb = self.z_proj(z_sample).view(B, 1, C, 1, 1)
        
        x_hist = x_hist + z_emb
        x_future = self.future_tokens.expand(B, -1, -1, H, W) + z_emb
        
        # 2. 序列拼接：[History, Future]
        x_seq = torch.cat([x_hist, x_future], dim=1) # [B, T_total, C, H, W]
        T_total = x_seq.shape[1]
        
        # 3. 叠加位置编码
        if T_total <= self.pos_embed_t.shape[1]:
            x_seq = x_seq + self.pos_embed_t[:, :T_total]
        
        # 4. Mamba 深度演变
        for layer in self.layers:
            x_seq = layer(x_seq)
            
        return x_seq[:, :T_in], x_seq[:, T_in:]

# ==============================================================================
# 主模型 (MeteoMamba)
# ==============================================================================

class MeteoMamba(nn.Module):
    """
    MeteoMamba V3 主模型架构。
    集成：Encoder -> CVAE -> Evolution(Mamba) + Advection -> Fusion -> Decoder
    """
    def __init__(self, 
                 in_shape, in_seq_len, out_seq_len, out_channels=1,
                 hid_S=64, hid_T=256, N_S=4, N_T=8,
                 spatio_kernel_enc=3, spatio_kernel_dec=3, 
                 mamba_d_state=16, mamba_d_conv=4, mamba_expand=2,
                 use_checkpoint=True,
                 **kwargs):
        super().__init__()
        
        C, H, W = in_shape
        self.T_in = in_seq_len
        self.T_out = out_seq_len
        self.out_channels = out_channels
        self.noise_dim = 32
        
        mamba_kwargs = {
            'd_state': mamba_d_state,
            'd_conv': mamba_d_conv,
            'expand': mamba_expand,
            'use_checkpoint': use_checkpoint
        }
        
        # 1. 编码器
        self.enc = Encoder(C, hid_S, N_S, spatio_kernel_enc)
        
        # 2. CVAE 网络
        # Prior P(z|X): 仅从历史预测 z (推理时使用)
        self.prior_net = PosteriorNet(hid_S, in_seq_len, self.noise_dim)
        # Posterior Q(z|X,Y): 从历史+未来预测 z (训练时使用)
        self.posterior_net = PosteriorNet(hid_S, in_seq_len + out_seq_len, self.noise_dim)
        
        # 3. 演变网络
        self.evolution = EvolutionNet(
            hid_S, N_T, 
            noise_dim=self.noise_dim,
            out_seq_len=out_seq_len,
            **mamba_kwargs
        )
        
        # 4. 物理平流分支
        self.advection = AdvectiveProjection(hid_S, in_seq_len, out_seq_len)
        
        # 5. 解码器与跳跃连接
        self.dec = Decoder(hid_S, out_channels, N_S, spatio_kernel_dec)
        self.skip_proj = TimeAlignBlock(in_seq_len, out_seq_len, hid_S)

    def reparameterize(self, mu, logvar):
        """重参数化技巧: z = mu + sigma * epsilon"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x_raw, y_target=None):
        """
        Args:
            x_raw: [B, T_in, C, H, W] 输入序列
            y_target: [B, T_out, C, H, W] 目标序列 (仅训练时提供，用于 CVAE 学习)
        Returns:
            Y_pred, flows, kl_loss (仅训练时)
        """
        B, T_in, C_in, H, W = x_raw.shape
        
        # 1. 编码阶段
        # [B, T, C, H, W] -> [B, T, hid_S, H', W']
        embed_hist, skip = self.enc(x_raw) 
        _, C_hid, H_, W_ = embed_hist.shape[1:]
        
        # 调整维度用于 3D 卷积: [B, C, T, H, W]
        embed_hist_seq = embed_hist.view(B, T_in, C_hid, H_, W_).permute(0, 2, 1, 3, 4)
        
        kl_loss = torch.tensor(0.0, device=x_raw.device)
        z = None
        
        # 2. CVAE 概率推断逻辑
        if self.training and y_target is not None:
            # === 训练模式 (Posterior 引导) ===
            with torch.no_grad():
                # 编码未来真值 (不传梯度，仅作参考)
                embed_future, _ = self.enc(y_target)
                embed_future_seq = embed_future.view(B, self.T_out, C_hid, H_, W_).permute(0, 2, 1, 3, 4)
            
            # Posterior Q(z|X,Y)
            embed_all = torch.cat([embed_hist_seq, embed_future_seq], dim=2)
            post_mu, post_logvar = self.posterior_net(embed_all)
            z = self.reparameterize(post_mu, post_logvar)
            
            # Prior P(z|X)
            prior_mu, prior_logvar = self.prior_net(embed_hist_seq)
            
            # KL Loss: 让 Posterior 逼近 Prior
            kl_loss = 0.5 * torch.sum(
                prior_logvar - post_logvar - 1 + 
                (post_logvar.exp() + (post_mu - prior_mu).pow(2)) / prior_logvar.exp()
            )
            kl_loss = kl_loss / B
            
        else:
            # === 推理模式 (Prior 采样) ===
            prior_mu, prior_logvar = self.prior_net(embed_hist_seq)
            z = self.reparameterize(prior_mu, prior_logvar)

        # 3. 演变阶段
        z_embed_in = embed_hist.view(B, T_in, C_hid, H_, W_)
        z_hist_out, z_future_out = self.evolution(z_embed_in, z)
        
        # 4. 物理平流阶段
        z_adv, flows = self.advection(z_hist_out)
        
        # 5. 特征融合
        z_combined = z_future_out + z_adv
        
        # 6. 解码阶段
        z_combined_flat = z_combined.reshape(B * self.T_out, C_hid, H_, W_)
        
        # Skip Connection 时间对齐
        skip = skip.view(B, T_in, C_hid, H, W)
        skip_out = self.skip_proj(skip).view(B * self.T_out, C_hid, H, W)
        
        Y_diff = self.dec(z_combined_flat, skip_out)
        Y_diff = Y_diff.reshape(B, self.T_out, self.out_channels, H, W)
        
        # 7. 残差输出
        last_frame = x_raw[:, -1:, :self.out_channels, :, :].detach()
        Y = last_frame + Y_diff
        
        if self.training:
            return Y, flows, kl_loss
        return Y, flows