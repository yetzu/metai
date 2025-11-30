# metai/model/met_mamba/model.py

import torch
import torch.nn as nn
from .modules import (
    ConvSC, 
    ResizeConv,
    STMambaBlock, 
    TimeAlignBlock
)

def sampling_generator(N, reverse=False):
    samplings = [False, True] * (N // 2)
    if reverse: return list(reversed(samplings[:N]))
    else: return samplings[:N]

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
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = self.stem(x) 
        enc1 = self.enc[0](x)
        latent = enc1
        for i in range(1, len(self.enc)):
            latent = self.enc[i](latent)
        return latent, enc1

class Decoder(nn.Module):
    """
    [改进] Decoder
    在最后两个上采样阶段使用 ResizeConv 替代 ConvSC 中的 PixelShuffle 或 TransposedConv，
    利用 "双线性插值 + 卷积" 消除棋盘格伪影 (Checkerboard Artifacts)，
    获得更平滑、自然的雷达回波预测结果。
    """
    def __init__(self, C_hid, C_out, N_S, spatio_kernel):
        super().__init__()
        samplings = sampling_generator(N_S, reverse=True)
        
        layers = []
        for i, s in enumerate(samplings):
            # 策略：前几层保持 ConvSC (计算效率高)，仅在最后两层(分辨率较高时)使用 ResizeConv
            if i < len(samplings) - 2:
                layers.append(ConvSC(C_hid, C_hid, spatio_kernel, upsampling=s))
            else:
                if s: # 需要上采样
                    layers.append(ResizeConv(C_hid, C_hid, spatio_kernel))
                else:
                    layers.append(ConvSC(C_hid, C_hid, spatio_kernel, upsampling=False))
                    
        self.dec = nn.Sequential(*layers)
        self.readout = nn.Conv2d(C_hid, C_out, 1)

    def forward(self, hid, skip=None):
        for i in range(0, len(self.dec)-1):
            hid = self.dec[i](hid)
        
        # Skip Connection fusion (Additive)
        if skip is not None:
            hid = hid + skip
            
        Y = self.dec[-1](hid)
        return self.readout(Y)

class EvolutionNet(nn.Module):
    """
    [改进] Evolution Network
    使用连续时空 STMambaBlock 替代原有的交替堆叠 (Spatial -> Temporal)。
    现在每一层都能同时进行时空状态的连续建模。
    """
    def __init__(self, dim_in, dim_hid, num_layers, drop=0., drop_path=0., mamba_kwargs={}):
        super().__init__()
        self.proj_in = nn.Linear(dim_in, dim_hid)
        self.proj_out = nn.Linear(dim_hid, dim_in)
        
        dpr = [x.item() for x in torch.linspace(1e-2, drop_path, num_layers)]
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            # 全部替换为 STMambaBlock
            self.layers.append(
                STMambaBlock(dim_hid, drop=drop, drop_path=dpr[i], **mamba_kwargs)
            )

    def forward(self, x):
        # x: [B, T, C, H, W]
        # 变换为 [B, T, H, W, C] 以适应 Linear 和 LayerNorm
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = self.proj_in(x)
        # 变换回 [B, T, C, H, W] 传递给 Block (STMambaBlock 内部会将 T 展开为序列)
        x = x.permute(0, 1, 4, 2, 3).contiguous()
        
        for layer in self.layers:
            x = layer(x)
            
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = self.proj_out(x)
        x = x.permute(0, 1, 4, 2, 3).contiguous()
        return x

class MeteoMamba(nn.Module):
    def __init__(self, 
                 in_shape,      # (C, H, W)
                 in_seq_len,    # T_in
                 out_seq_len,   # T_out
                 out_channels=1,
                 hid_S=64, 
                 hid_T=256, 
                 N_S=4, 
                 N_T=8, 
                 spatio_kernel_enc=3, 
                 spatio_kernel_dec=3, 
                 mamba_d_state=16, 
                 mamba_d_conv=4, 
                 mamba_expand=2,
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
        
        self.enc = Encoder(C, hid_S, N_S, spatio_kernel_enc)
        
        # 核心演变网络: 使用 STMambaBlock 堆叠
        self.evolution = EvolutionNet(
            hid_S, hid_T, N_T, 
            drop=0.0, drop_path=0.1, 
            mamba_kwargs=mamba_kwargs
        )
        
        # 潜在空间时间投影 (T_in -> T_out)
        self.latent_time_proj = nn.Sequential(
            nn.Conv1d(T_in, self.T_out, kernel_size=1),
            nn.Conv1d(self.T_out, self.T_out, kernel_size=3, padding=1, groups=1),
            nn.SiLU()
        )
        
        self.dec = Decoder(hid_S, self.out_channels, N_S, spatio_kernel_dec)
        self.skip_proj = TimeAlignBlock(T_in, self.T_out, hid_S)
        self._init_time_proj()

    def _init_time_proj(self):
        for m in self.latent_time_proj.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, x_raw):
        # x_raw: [B, T_in, C_in, H, W]
        B, T_in, C_in, H, W = x_raw.shape
        
        # 1. Encoder: 提取空间特征 -> [B*T_in, C_hid, H_, W_]
        embed, skip = self.enc(x_raw) 
        _, C_hid, H_, W_ = embed.shape 
        
        # 2. Evolution: 时空演变 (STMamba) -> [B, T_in, C_hid, H_, W_]
        z = embed.view(B, T_in, C_hid, H_, W_)
        z = self.evolution(z)
        
        # 3. Time Projection: 预测未来时间步 -> [B, T_out, C_hid, H_, W_]
        z = z.reshape(B, T_in, -1) # [B, T_in, C*H*W]
        z = self.latent_time_proj(z) # [B, T_out, C*H*W]
        z = z.reshape(B * self.T_out, C_hid, H_, W_)
        
        # 4. Skip Connection 处理
        skip = skip.view(B, T_in, C_hid, H, W)
        skip_out = self.skip_proj(skip) # [B, T_out, C_hid, H, W]
        skip_out = skip_out.view(B * self.T_out, C_hid, H, W)
        
        # 5. Decoder: 还原空间分辨率 (含 ResizeConv) -> [B, T_out, 1, H, W]
        Y_diff = self.dec(z, skip_out)
        Y_diff = Y_diff.reshape(B, self.T_out, self.out_channels, H, W)
        
        # 6. Residual Connection: 加上最后一帧观测作为基准
        last_frame = x_raw[:, -1:, :self.out_channels, :, :].detach() 
        Y = last_frame + Y_diff
        
        return Y