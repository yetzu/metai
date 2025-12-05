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
    samplings = [False, True] * (N // 2)
    if reverse: return list(reversed(samplings[:N]))
    else: return samplings[:N]

# ==============================================================================
# 编码器与解码器 (Encoder & Decoder) - 保持不变
# ==============================================================================

class Encoder(nn.Module):
    def __init__(self, C_in, C_hid, N_S, spatio_kernel):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(C_in, C_hid, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(2, C_hid),
            nn.SiLU(inplace=True)
        )
        samplings = sampling_generator(N_S)
        self.enc = nn.Sequential(
            ConvSC(C_hid, C_hid, spatio_kernel, downsampling=samplings[0]),
            *[ConvSC(C_hid, C_hid, spatio_kernel, downsampling=s) for s in samplings[1:]]
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = self.stem(x) 
        enc1 = self.enc[0](x)
        latent = enc1
        for i in range(1, len(self.enc)):
            latent = self.enc[i](latent)
        return latent, enc1


class Decoder(nn.Module):
    def __init__(self, C_hid, C_out, N_S, spatio_kernel):
        super().__init__()
        samplings = sampling_generator(N_S, reverse=True)
        layers = []
        for i, s in enumerate(samplings):
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
        for i in range(0, len(self.dec)-1):
            hid = self.dec[i](hid)
        if skip is not None:
            hid = hid + skip
        Y = self.dec[-1](hid)
        return self.readout(Y)

# ==============================================================================
# 演变网络 (EvolutionNet) - [REFACTORED]
# ==============================================================================

class EvolutionNet(nn.Module):
    """
    [改进版 v2.0] 全序列自回归演变网络
    特性：
    1. 引入 future_tokens，将序列扩展为 T_in + T_out。
    2. Mamba 负责在隐空间内进行全序列的非线性生消演变。
    """
    def __init__(self, dim_in, dim_hid, num_layers, drop=0., drop_path=0., 
                 mamba_kwargs={}, use_checkpoint=True, max_t=64, input_resolution=(64, 64),
                 sparse_ratio=0.0,
                 anneal_start_epoch=5, anneal_end_epoch=10,
                 out_seq_len=20): # [NEW] 增加输出序列长度参数
        super().__init__()
        
        # 1. 噪声投影层
        self.noise_dim = 32
        self.noise_proj = nn.Linear(self.noise_dim, dim_hid) 
        nn.init.zeros_(self.noise_proj.weight)
        nn.init.zeros_(self.noise_proj.bias)
        
        # 2. 特征投影
        self.proj_in = nn.Linear(dim_in, dim_hid)
        self.proj_out = nn.Linear(dim_hid, dim_in)
        
        # [NEW] 3. 未来查询向量 (Learnable Query Tokens)
        # 用于填补未来 T_out 的空缺，作为 Mamba 预测未来的载体
        self.future_tokens = nn.Parameter(torch.zeros(1, out_seq_len, dim_hid, 1, 1))
        trunc_normal_(self.future_tokens, std=0.02)
        
        # 4. 位置编码 (Time & Space PE)
        # 扩大 max_t 以支持更长的预测序列
        max_t = max(max_t, 32 + out_seq_len) 
        self.pos_embed_t = nn.Parameter(torch.zeros(1, max_t, dim_hid, 1, 1)) 
        H_feat, W_feat = input_resolution
        self.pos_embed_s = nn.Parameter(torch.zeros(1, 1, dim_hid, H_feat, W_feat)) 
        self.time_prompt = nn.Parameter(torch.zeros(1, max_t, dim_hid, 1, 1))

        trunc_normal_(self.pos_embed_t, std=0.02)
        trunc_normal_(self.pos_embed_s, std=0.02)
        trunc_normal_(self.time_prompt, std=0.02)
        
        # 5. 堆叠 STMamba Block
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
            x: [B, T_in, C, H, W] 历史特征
        Returns:
            z_history: [B, T_in, C, H, W] 演变后的历史特征
            z_future:  [B, T_out, C, H, W] 演变出的未来特征
        """
        B, T_in, C, H, W = x.shape
        T_out = self.future_tokens.shape[1]
        
        # 1. 噪声生成 (仅注入历史部分，或者也可以注入未来，这里保持原逻辑注入输入)
        if self.training:
            noise = torch.randn(B, T_in, self.noise_dim, device=x.device)
        else:
            noise = torch.zeros(B, T_in, self.noise_dim, device=x.device)
        noise_emb = self.noise_proj(noise).unsqueeze(3).unsqueeze(3)
        
        # 2. 特征投影
        x = x.permute(0, 1, 3, 4, 2).contiguous() # [B, T, H, W, C]
        x = self.proj_in(x)
        x = x.permute(0, 1, 4, 2, 3).contiguous() # [B, T, C_hid, H, W]
        
        # 3. 融合噪声
        x = x + noise_emb
        
        # [NEW] 4. 序列拼接 (Concatenate History + Future Query)
        # 扩展 future_tokens 到当前 Batch
        future_queries = self.future_tokens.expand(B, -1, -1, H, W)
        x_seq = torch.cat([x, future_queries], dim=1) # [B, T_total, C_hid, H, W]
        
        T_total = x_seq.shape[1]
        
        # 5. 位置编码插值与叠加
        # Time PE (适配 T_total)
        if T_total <= self.time_prompt.shape[1]:
            t_prompt = self.time_prompt[:, :T_total, ...]
            t_pos = self.pos_embed_t[:, :T_total, ...]
        else:
            t_prompt = F.interpolate(
                self.time_prompt.squeeze(-1).squeeze(-1).permute(0,2,1), 
                size=T_total, mode='linear'
            ).permute(0,2,1).unsqueeze(-1).unsqueeze(-1)
            t_pos = F.interpolate(
                self.pos_embed_t.squeeze(-1).squeeze(-1).permute(0,2,1), 
                size=T_total, mode='linear'
            ).permute(0,2,1).unsqueeze(-1).unsqueeze(-1)

        # Spatial PE
        if (H, W) != self.pos_embed_s.shape[-2:]:
            s_pos = F.interpolate(
                self.pos_embed_s, size=(H, W), mode='bilinear', align_corners=False
            )
        else:
            s_pos = self.pos_embed_s

        x_seq = x_seq + t_prompt + t_pos + s_pos
        
        # 6. 通过 STMamba 层 (一次性处理历史+未来)
        for layer in self.layers:
            x_seq = layer(x_seq, current_epoch=current_epoch)
            
        # 7. 输出投影
        x_seq = x_seq.permute(0, 1, 3, 4, 2).contiguous()
        x_seq = self.proj_out(x_seq)
        x_seq = x_seq.permute(0, 1, 4, 2, 3).contiguous()
        
        # 8. 分离历史与未来
        z_history = x_seq[:, :T_in]
        z_future = x_seq[:, T_in:] 
        
        return z_history, z_future

# ==============================================================================
# 主模型 (Main Model Class) - [REFACTORED]
# ==============================================================================

class MeteoMamba(nn.Module):
    """
    [MeteoMamba v2.0 Final] 
    1. Encoder: 提取空间特征
    2. EvolutionNet: Autoregressive Mamba (全序列推演)
    3. Advection: 物理约束 (作为 Residual 分支)
    4. Decoder: 特征恢复
    """
    def __init__(self, 
                 in_shape, in_seq_len, out_seq_len, out_channels=1,
                 hid_S=64, hid_T=256, N_S=4, N_T=8,
                 spatio_kernel_enc=3, spatio_kernel_dec=3, 
                 mamba_d_state=16, mamba_d_conv=4, mamba_expand=2,
                 use_checkpoint=True,
                 mamba_sparse_ratio=0.5, 
                 anneal_start_epoch=5, anneal_end_epoch=10,
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
        
        ds_factor = 2 ** (N_S // 2) 
        evo_res = (H // ds_factor, W // ds_factor)

        # 2. 初始化演变网络 (传入 out_seq_len)
        self.evolution = EvolutionNet(
            hid_S, hid_T, N_T, 
            drop=0.0, drop_path=0.1, 
            mamba_kwargs=mamba_kwargs,
            use_checkpoint=use_checkpoint,
            input_resolution=evo_res,
            sparse_ratio=mamba_sparse_ratio,
            anneal_start_epoch=anneal_start_epoch, 
            anneal_end_epoch=anneal_end_epoch,
            out_seq_len=out_seq_len  # [NEW]
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

    def forward(self, x_raw, current_epoch=0):
        """
        Returns:
            Y: [B, T_out, C_out, H, W]
            flows: [B, T_out, 2, H, W]
        """
        B, T_in, C_in, H, W = x_raw.shape
        
        # 1. 编码阶段
        embed, skip = self.enc(x_raw) 
        _, C_hid, H_, W_ = embed.shape 
        
        # 2. 演变阶段 (Mamba Autoregressive)
        # z_hist: 历史特征演变, z_future_mamba: Mamba预测的未来特征(含生消)
        z = embed.view(B, T_in, C_hid, H_, W_)
        z_hist, z_future_mamba = self.evolution(z, current_epoch=current_epoch)
        
        # 3. 物理平流阶段 (Advection Constraint)
        # 基于历史最后一帧 z_hist[:, -1] 递归推导纯平流预测 z_adv
        z_adv, flows = self.latent_time_proj(z_hist) 
        
        # 4. 物理-深度残差融合 (Physics-Residual Fusion)
        # 最终特征 = Mamba预测(非线性变化) + Advection预测(平流惯性)
        z_combined = z_future_mamba + z_adv
        
        # 5. 解码准备
        z_combined = z_combined.view(B * self.T_out, C_hid, H_, W_)
        
        # 6. 处理跳跃连接
        skip = skip.view(B, T_in, C_hid, H, W)
        skip_out = self.skip_proj(skip) 
        skip_out = skip_out.view(B * self.T_out, C_hid, H, W)
        
        # 7. 解码阶段
        Y_diff = self.dec(z_combined, skip_out)
        Y_diff = Y_diff.reshape(B, self.T_out, self.out_channels, H, W)
        
        # 8. 残差连接 (预测增量)
        last_frame = x_raw[:, -1:, :self.out_channels, :, :].detach() 
        Y = last_frame + Y_diff
        
        return Y, flows