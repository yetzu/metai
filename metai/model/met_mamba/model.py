# metai/model/met_mamba/model.py

import torch
import torch.nn as nn
from .modules import ConvSC, SpatialMambaBlock, TemporalMambaBlock, TimeAlignBlock

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
    """
    Evolution Network with Interleaved Stacking:
    Spatial (SS2D) -> Temporal -> Spatial (SS2D) -> Temporal ...
    """
    def __init__(self, dim_in, dim_hid, num_layers, drop=0., drop_path=0., mamba_kwargs={}):
        super().__init__()
        self.proj_in = nn.Linear(dim_in, dim_hid)
        self.proj_out = nn.Linear(dim_hid, dim_in)
        
        dpr = [x.item() for x in torch.linspace(1e-2, drop_path, num_layers)]
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            # 交替构建
            if i % 2 == 0:
                self.layers.append(
                    SpatialMambaBlock(dim_hid, drop=drop, drop_path=dpr[i], **mamba_kwargs)
                )
            else:
                self.layers.append(
                    TemporalMambaBlock(dim_hid, drop=drop, drop_path=dpr[i], **mamba_kwargs)
                )

    def forward(self, x):
        # x: [B, T, C, H, W]
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = self.proj_in(x)
        x = x.permute(0, 1, 4, 2, 3).contiguous()
        
        for layer in self.layers:
            # [Optimization] 直接前向传播，不使用 checkpoint
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
        self.evolution = EvolutionNet(hid_S, hid_T, N_T, drop=0.0, drop_path=0.1, mamba_kwargs=mamba_kwargs)
        
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
        B, T_in, C_in, H, W = x_raw.shape
        
        embed, skip = self.enc(x_raw) 
        _, C_hid, H_, W_ = embed.shape 
        
        z = embed.view(B, T_in, C_hid, H_, W_)
        z = self.evolution(z) 
        
        z = z.reshape(B, T_in, -1) 
        z = self.latent_time_proj(z)
        z = z.reshape(B * self.T_out, C_hid, H_, W_)
        
        skip = skip.view(B, T_in, C_hid, H, W)
        skip_out = self.skip_proj(skip)
        skip_out = skip_out.view(B * self.T_out, C_hid, H, W)
        
        Y_diff = self.dec(z, skip_out)
        Y_diff = Y_diff.reshape(B, self.T_out, self.out_channels, H, W)
        
        last_frame = x_raw[:, -1:, :self.out_channels, :, :].detach() 
        Y = last_frame + Y_diff
        
        return Y