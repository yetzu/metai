import torch
import torch.nn as nn
from .modules import ConvSC, STMambaBlock, TimeAlignBlock

def sampling_generator(N, reverse=False):
    samplings = [False, True] * (N // 2)
    if reverse: return list(reversed(samplings[:N]))
    else: return samplings[:N]

class Encoder(nn.Module):
    """
    [精简版] Pure 2D Encoder
    专注于提取单帧的空间特征，移除冗余的 3D 卷积。
    """
    def __init__(self, C_in, C_hid, N_S, spatio_kernel):
        super().__init__()
        
        # [修改] 2D Stem: 只在空间维度处理，大幅降低计算量
        # C_in -> C_hid
        self.stem = nn.Sequential(
            nn.Conv2d(C_in, C_hid, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(2, C_hid), # GroupNorm 依然适用
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
        
        # Flatten time into batch: [B*T, C, H, W]
        x = x.view(B * T, C, H, W)
        
        # 2D Spatial Encoding
        x = self.stem(x) 
        
        # ConvSC Stages
        enc1 = self.enc[0](x)
        latent = enc1
        for i in range(1, len(self.enc)):
            latent = self.enc[i](latent)
            
        # latent: [B*T, C_hid, H', W']
        # enc1:   [B*T, C_hid, H, W]
        return latent, enc1

class Decoder(nn.Module):
    """Pure 2D Decoder"""
    def __init__(self, C_hid, C_out, N_S, spatio_kernel):
        super().__init__()
        samplings = sampling_generator(N_S, reverse=True)
        self.dec = nn.Sequential(
            *[ConvSC(C_hid, C_hid, spatio_kernel, upsampling=s) for s in samplings[:-1]],
              ConvSC(C_hid, C_hid, spatio_kernel, upsampling=samplings[-1])
        )
        self.readout = nn.Conv2d(C_hid, C_out, 1)

    def forward(self, hid, skip=None):
        for i in range(0, len(self.dec)-1):
            hid = self.dec[i](hid)
        Y = self.dec[-1](hid + skip)
        return self.readout(Y)

class EvolutionNet(nn.Module):
    """Mamba-based Evolution Network (保持不变，这是核心)"""
    def __init__(self, dim_in, dim_hid, num_layers, drop=0., drop_path=0.):
        super().__init__()
        self.proj_in = nn.Linear(dim_in, dim_hid)
        self.proj_out = nn.Linear(dim_hid, dim_in)
        dpr = [x.item() for x in torch.linspace(1e-2, drop_path, num_layers)]
        self.layers = nn.ModuleList([
            STMambaBlock(dim_hid, drop=drop, drop_path=dpr[i]) for i in range(num_layers)
        ])

    def forward(self, x):
        # x: [B, T, C, H, W]
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = self.proj_in(x)
        x = x.permute(0, 1, 4, 2, 3).contiguous()
        for layer in self.layers:
            x = layer(x)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = self.proj_out(x)
        x = x.permute(0, 1, 4, 2, 3).contiguous()
        return x

class MeteoMamba(nn.Module):
    def __init__(self, in_shape, hid_S=64, hid_T=256, N_S=4, N_T=8, 
                 spatio_kernel_enc=3, spatio_kernel_dec=3, 
                 out_channels=None, aft_seq_length=20, **kwargs):
        super().__init__()
        T_in, C, H, W = in_shape
        self.T_out = aft_seq_length
        if out_channels is None: out_channels = C
        self.out_channels = out_channels
        
        # 1. Pure 2D Encoder
        self.enc = Encoder(C, hid_S, N_S, spatio_kernel_enc)
        
        # 2. Spatiotemporal Evolution
        self.evolution = EvolutionNet(hid_S, hid_T, N_T, drop=0.0, drop_path=0.1)
        
        # 3. Temporal Extrapolation (Conv1d)
        self.latent_time_proj = nn.Sequential(
            nn.Conv1d(T_in, self.T_out, kernel_size=1),
            nn.Conv1d(self.T_out, self.T_out, kernel_size=3, padding=1, groups=1),
            nn.SiLU()
        )
        
        # 4. Decoder
        self.dec = Decoder(hid_S, self.out_channels, N_S, spatio_kernel_dec)
        
        # 5. Skip Connection Projection
        self.skip_proj = TimeAlignBlock(T_in, self.T_out, hid_S)
        
        self._init_time_proj()

    def _init_time_proj(self):
        for m in self.latent_time_proj.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, x_raw):
        # x_raw: [B, T_in, C, H, W]
        B, T_in, C_in, H, W = x_raw.shape
        
        # --- 1. Encoding (Spatial Only) ---
        embed, skip = self.enc(x_raw) 
        # embed: [B*T_in, C_hid, H', W']
        
        _, C_hid, H_, W_ = embed.shape 
        
        # --- 2. Evolution (Spatiotemporal) ---
        z = embed.view(B, T_in, C_hid, H_, W_)
        z = self.evolution(z) # [B, T_in, C_hid, H_, W_]
        
        # --- 3. Extrapolation (Temporal) ---
        z = z.permute(0, 2, 3, 4, 1).contiguous().view(B, C_hid * H_ * W_, T_in)
        z = self.latent_time_proj(z) # -> [B, ..., T_out]
        z = z.view(B, C_hid, H_, W_, self.T_out).permute(0, 4, 1, 2, 3).contiguous()
        # z: [B, T_out, C_hid, H_, W_] -> Flatten
        z = z.view(B * self.T_out, C_hid, H_, W_)
        
        # --- 4. Skip Connection ---
        skip = skip.view(B, T_in, C_hid, H, W)
        skip_out = self.skip_proj(skip)
        skip_out = skip_out.view(B * self.T_out, C_hid, H, W)
        
        # --- 5. Decoding ---
        Y_diff = self.dec(z, skip_out)
        Y_diff = Y_diff.reshape(B, self.T_out, self.out_channels, H, W)
        
        # [核心策略改进] 残差学习 (Residual Learning)
        # 模型预测的是变化量 Y_diff，最终结果 = 最后一帧 + 变化量
        # 注意：这要求 T_out 序列是累积的，这里我们简化为：
        # 每一帧的预测基于上一帧（但这在并行预测中较难），或者：
        # 将 Y_diff 视为相对于 Input 最后一帧的变化。
        
        last_frame = x_raw[:, -1:].detach() # [B, 1, C, H, W]
        Y = last_frame + Y_diff
        
        return Y