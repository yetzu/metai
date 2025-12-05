# metai/model/met_mamba/model.py

import torch
import torch.nn as nn
from timm.layers.weight_init import trunc_normal_

# 引用基础组件
from .modules import (
    ConvSC, 
    ResizeConv, 
    STMambaBlock, 
    TimeAlignBlock, 
    AdvectiveProjection, 
    PosteriorNet
)

# ==============================================================================
# 编码器 (Encoder)
# ==============================================================================

class Encoder(nn.Module):
    """
    空间编码器：将高分辨率雷达观测序列压缩为低维隐特征。
    结构：Stem 卷积 -> 多层下采样卷积块 (ConvSC)
    """
    def __init__(self, C_in, C_hid, N_S, spatio_kernel):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(C_in, C_hid, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(2, C_hid),
            nn.SiLU(inplace=True)
        )
        
        # 生成下采样控制序列 (例如: [False, True, False, True])
        # N_S 控制编码器的深度
        samplings = [False, True] * (N_S // 2)
        
        layers = []
        # 第一层
        layers.append(ConvSC(C_hid, C_hid, spatio_kernel, downsampling=samplings[0]))
        # 后续层
        for s in samplings[1:]:
            layers.append(ConvSC(C_hid, C_hid, spatio_kernel, downsampling=s))
            
        self.enc = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: [B, T, C, H, W] 输入序列
        Returns:
            latent: [B*T, C_hid, H', W'] 深层特征
            enc1: [B*T, C_hid, H, W] 浅层特征 (用于 Skip Connection)
        """
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        
        x = self.stem(x) 
        enc1 = self.enc[0](x) # 保留浅层特征
        
        latent = enc1
        for i in range(1, len(self.enc)):
            latent = self.enc[i](latent)
            
        return latent, enc1

# ==============================================================================
# 解码器 (Decoder)
# ==============================================================================

class Decoder(nn.Module):
    """
    空间解码器：将演变后的隐特征恢复为预测图。
    结构：多层上采样卷积块 -> 输出投影
    """
    def __init__(self, C_hid, C_out, N_S, spatio_kernel):
        super().__init__()
        # 生成上采样控制序列 (与 Encoder 相反)
        samplings = list(reversed([False, True] * (N_S // 2)))
        
        layers = []
        for i, s in enumerate(samplings):
            if i < len(samplings) - 2:
                layers.append(ConvSC(C_hid, C_hid, spatio_kernel, upsampling=s))
            else:
                # 倒数第二层特殊处理 (ResizeConv 或 ConvSC)
                if s: 
                    layers.append(ResizeConv(C_hid, C_hid, spatio_kernel))
                else: 
                    layers.append(ConvSC(C_hid, C_hid, spatio_kernel, upsampling=False))
                    
        self.dec = nn.Sequential(*layers)
        self.readout = nn.Conv2d(C_hid, C_out, 1)

    def forward(self, hid, skip=None):
        """
        Args:
            hid: [B*T, C_hid, H', W'] 隐特征
            skip: [B*T, C_hid, H, W] 跳跃连接特征 (可选)
        Returns:
            Y: [B*T, C_out, H, W] 重建结果
        """
        # 解码主体 (除最后一层)
        for i in range(0, len(self.dec)-1):
            hid = self.dec[i](hid)
            
        # 融合 Skip Connection
        if skip is not None:
            hid = hid + skip
            
        # 最后一层解码 + Readout
        Y = self.dec[-1](hid)
        return self.readout(Y)

# ==============================================================================
# 演变网络 (EvolutionNet)
# ==============================================================================

class EvolutionNet(nn.Module):
    """
    时空演变网络核心。
    功能：结合 CVAE 隐变量 z，利用 STMamba 在隐空间推演未来特征。
    """
    def __init__(self, dim, num_layers, out_seq_len, noise_dim=32, drop_path=0.1, **kwargs):
        super().__init__()
        self.noise_dim = noise_dim
        
        # 1. 噪声投影层：将隐变量 z 映射到特征维度
        self.z_proj = nn.Linear(noise_dim, dim)
        
        # 2. 未来查询向量 (Learned Queries)：承载未来预测状态的容器
        # 相比简单的零填充，可学习的 Query 能捕捉平均气候态
        self.future_tokens = nn.Parameter(torch.zeros(1, out_seq_len, dim, 1, 1))
        trunc_normal_(self.future_tokens, std=0.02)
        
        # 3. 可学习的时间位置编码
        self.pos_embed_t = nn.Parameter(torch.zeros(1, 64, dim, 1, 1)) 
        trunc_normal_(self.pos_embed_t, std=0.02)
        
        # 4. 堆叠 STMamba Block (Bi-2D-Mamba + Soft-Gating)
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
        
        # 1. 注入噪声 (Conditioning on z)
        # z 被广播到全时空，代表本次预测的全局随机扰动 (Global Stochasticity)
        z_emb = self.z_proj(z_sample).view(B, 1, C, 1, 1)
        
        x_hist = x_hist + z_emb
        x_future = self.future_tokens.expand(B, -1, -1, H, W) + z_emb
        
        # 2. 序列拼接：[History, Future]
        x_seq = torch.cat([x_hist, x_future], dim=1) # [B, T_total, C, H, W]
        T_total = x_seq.shape[1]
        
        # 3. 叠加位置编码 (广播到 Spatial 维度)
        if T_total <= self.pos_embed_t.shape[1]:
            x_seq = x_seq + self.pos_embed_t[:, :T_total]
        
        # 4. Mamba 深度演变 (In-context Learning)
        for layer in self.layers:
            x_seq = layer(x_seq)
            
        # 分离历史与未来部分
        return x_seq[:, :T_in], x_seq[:, T_in:]

# ==============================================================================
# 主模型 (MeteoMamba)
# ==============================================================================

class MeteoMamba(nn.Module):
    """
    MeteoMamba 主模型架构。
    
    设计理念：
    1. CVAE: 提供概率生成能力，防止 0 值坍塌 (Zero Collapse)。
    2. Advection: 物理平流提供保底预测，防止第二帧迅速衰减。
    3. Mamba: 线性复杂度捕捉长程时空依赖。
    4. Fusion: 非线性融合层动态结合深度学习预测与物理平流预测。
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
        # Prior P(z|X): 推理时使用，仅依赖历史
        self.prior_net = PosteriorNet(hid_S, latent_dim=self.noise_dim)
        # Posterior Q(z|X,Y): 训练时使用，依赖历史+未来
        self.posterior_net = PosteriorNet(hid_S, latent_dim=self.noise_dim)
        
        # 3. 演变网络 (Mamba)
        self.evolution = EvolutionNet(
            hid_S, N_T, 
            noise_dim=self.noise_dim,
            out_seq_len=out_seq_len,
            **mamba_kwargs
        )
        
        # 4. 物理平流分支 (Advection)
        self.advection = AdvectiveProjection(hid_S, in_seq_len, out_seq_len)
        
        # 5. 特征融合层 (Non-linear Fusion)
        # 输入: z_future_out (hid_S) + z_adv (hid_S) = 2 * hid_S
        # 输出: hid_S
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(hid_S * 2, hid_S, kernel_size=1),
            nn.GroupNorm(8, hid_S),
            nn.SiLU(inplace=True)
        )
        
        # 6. 解码器与跳跃连接
        self.dec = Decoder(hid_S, out_channels, N_S, spatio_kernel_dec)
        self.skip_proj = TimeAlignBlock(in_seq_len, out_seq_len, hid_S)

    def reparameterize(self, mu, logvar):
        """重参数化技巧: z = mu + sigma * epsilon"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x_raw, y_target=None):
        """
        前向传播。
        
        Args:
            x_raw: [B, T_in, C, H, W] 输入历史序列
            y_target: [B, T_out, C, H, W] 目标未来序列 (仅训练时提供)
            
        Returns:
            Y_pred: 预测结果
            flows: 光流场 (用于可视化或分析)
            kl_loss: KL 散度损失 (仅在训练模式下返回非零值)
        """
        B, T_in, C_in, H, W = x_raw.shape
        
        # 1. 编码阶段
        # embed_hist: [B*T_in, hid_S, H', W']
        # skip: [B*T_in, hid_S, H, W]
        embed_hist, skip = self.enc(x_raw) 
        
        # [Fix] 修正解包错误：embed_hist.shape[1:] 只有3个元素 (C, H, W)，只需3个变量
        C_hid, H_, W_ = embed_hist.shape[1:]
        
        # 调整维度用于 3D 卷积处理: [B, C, T, H, W]
        embed_hist_seq = embed_hist.view(B, T_in, C_hid, H_, W_).permute(0, 2, 1, 3, 4)
        
        kl_loss = torch.tensor(0.0, device=x_raw.device)
        z = None
        
        # 2. CVAE 概率推断逻辑
        if self.training and y_target is not None:
            # === 训练模式 (Posterior Guided) ===
            # 需要"看到未来"来构建后验分布 Q(z|X,Y)
            with torch.no_grad():
                # 编码未来真值 (不传梯度，仅作为分布参考)
                embed_future, _ = self.enc(y_target)
                embed_future_seq = embed_future.view(B, self.T_out, C_hid, H_, W_).permute(0, 2, 1, 3, 4)
            
            # 拼接历史与未来
            embed_all = torch.cat([embed_hist_seq, embed_future_seq], dim=2)
            
            # 计算 Posterior Q
            post_mu, post_logvar = self.posterior_net(embed_all)
            z = self.reparameterize(post_mu, post_logvar)
            
            # 计算 Prior P (仅基于历史)
            prior_mu, prior_logvar = self.prior_net(embed_hist_seq)
            
            # 计算 KL(Q || P)
            # 迫使 Posterior 分布接近 Prior 分布，缩小 Training/Inference Gap
            kl_loss = 0.5 * torch.sum(
                prior_logvar - post_logvar - 1 + 
                (post_logvar.exp() + (post_mu - prior_mu).pow(2)) / prior_logvar.exp()
            )
            kl_loss = kl_loss / B  # Normalize by batch size
            
        else:
            # === 推理模式 (Prior Sampling) ===
            # 仅使用 P(z|X) 从历史推断 z
            prior_mu, prior_logvar = self.prior_net(embed_hist_seq)
            z = self.reparameterize(prior_mu, prior_logvar)

        # 3. 演变阶段
        # 恢复 [B, T, C, H, W] 格式
        z_embed_in = embed_hist.view(B, T_in, C_hid, H_, W_)
        z_hist_out, z_future_out = self.evolution(z_embed_in, z)
        
        # 4. 物理平流分支 (Advection)
        # 基于历史最后一帧推演纯物理移动
        z_adv, flows = self.advection(z_hist_out)
        
        # 5. 特征融合 (Deep Learning + Physics)
        # 使用非线性卷积融合替代原有的 z_combined = z_future_out + z_adv
        
        # Reshape to [B*T_out, C, H, W] for 2D Conv
        z_future_flat = z_future_out.reshape(B * self.T_out, C_hid, H_, W_)
        z_adv_flat = z_adv.reshape(B * self.T_out, C_hid, H_, W_)
        
        # Concat along channel
        z_cat = torch.cat([z_future_flat, z_adv_flat], dim=1) # -> [B*T, 2*C, H, W]
        
        # Apply fusion
        z_combined_flat = self.fusion_conv(z_cat) # -> [B*T, C, H, W]
        
        # 6. 解码阶段
        # Skip Connection 时间投影
        skip = skip.view(B, T_in, C_hid, H, W)
        skip_out = self.skip_proj(skip).view(B * self.T_out, C_hid, H, W)
        
        # 解码
        Y_diff = self.dec(z_combined_flat, skip_out)
        Y_diff = Y_diff.reshape(B, self.T_out, self.out_channels, H, W)
        
        # 7. 残差输出
        # 最终预测 = 最后一帧观测 + 预测的变化量
        last_frame = x_raw[:, -1:, :self.out_channels, :, :].detach()
        Y = last_frame + Y_diff
        
        if self.training:
            return Y, flows, kl_loss
        return Y, flows