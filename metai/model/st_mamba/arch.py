import torch
import torch.nn as nn
from .modules import ConvSC, STMambaBlock, TimeAlignBlock

def sampling_generator(N, reverse=False):
    samplings = [False, True] * (N // 2)
    if reverse: return list(reversed(samplings[:N]))
    else: return samplings[:N]

class FeatureEncoder(nn.Module):
    def __init__(self, C_in, C_hid, N_S, spatio_kernel):
        super().__init__()
        samplings = sampling_generator(N_S)
        self.enc = nn.Sequential(
              ConvSC(C_in, C_hid, spatio_kernel, downsampling=samplings[0]),
            *[ConvSC(C_hid, C_hid, spatio_kernel, downsampling=s) for s in samplings[1:]]
        )

    def forward(self, x):
        # x: [B*T, C, H, W]
        enc1 = self.enc[0](x)
        latent = enc1
        for i in range(1, len(self.enc)):
            latent = self.enc[i](latent)
        return latent, enc1

class FeatureDecoder(nn.Module):
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
    def __init__(self, dim_in, dim_hid, num_layers, drop=0., drop_path=0.):
        super().__init__()
        self.proj_in = nn.Linear(dim_in, dim_hid)
        self.proj_out = nn.Linear(dim_hid, dim_in)
        
        dpr = [x.item() for x in torch.linspace(1e-2, drop_path, num_layers)]
        self.layers = nn.ModuleList([
            STMambaBlock(dim_hid, drop=drop, drop_path=dpr[i]) 
            for i in range(num_layers)
        ])

    def forward(self, x):
        # x: [B, T, C, H, W]
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = self.proj_in(x) # Project Channel
        
        # [B, T, H, W, C_hid] -> STMambaBlock expects [B, T, C, H, W]
        x = x.permute(0, 1, 4, 2, 3).contiguous()
        
        for layer in self.layers:
            x = layer(x)
            
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = self.proj_out(x) # Project Back
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
        
        # 1. Encoder
        self.enc = FeatureEncoder(C, hid_S, N_S, spatio_kernel_enc)
        
        # 2. Evolution
        self.evolution = EvolutionNet(hid_S, hid_T, N_T, drop=0.0, drop_path=0.1)
        
        # 3. Temporal Extrapolation
        self.latent_time_proj = nn.Linear(T_in, self.T_out)
        
        # 4. Decoder (Corrected: output channels should be self.out_channels)
        self.dec = FeatureDecoder(hid_S, self.out_channels, N_S, spatio_kernel_dec)
        
        # 5. Skip Connection Extrapolation
        self.skip_proj = TimeAlignBlock(T_in, self.T_out, hid_S)
        
        self._init_time_proj()

    def _init_time_proj(self):
        nn.init.normal_(self.latent_time_proj.weight, mean=0.0, std=0.01)
        with torch.no_grad():
            if self.latent_time_proj.weight.shape[1] > 0:
                self.latent_time_proj.weight[:, -1].fill_(1.0)
        nn.init.constant_(self.latent_time_proj.bias, 0)

    def forward(self, x_raw):
        # x_raw: [B, T_in, C, H, W]
        B, T_in, C_in, H, W = x_raw.shape
        x = x_raw.view(B*T_in, C_in, H, W)

        # Encode
        embed, skip = self.enc(x)
        _, C_hid, H_, W_ = embed.shape 
        
        # Evolve & Extrapolate
        z = embed.view(B, T_in, C_hid, H_, W_)
        z = self.evolution(z) # [B, T_in, C_hid, H', W']
        
        # Temporal Project
        z = z.permute(0, 2, 3, 4, 1) # [B, C, H, W, T_in]
        z = self.latent_time_proj(z) # [B, C, H, W, T_out]
        z = z.permute(0, 4, 1, 2, 3).contiguous() 
        z = z.view(B * self.T_out, C_hid, H_, W_)
        
        # Skip Connection
        skip = skip.view(B, T_in, C_hid, H, W)
        skip_out = self.skip_proj(skip)
        skip_out = skip_out.view(B * self.T_out, C_hid, H, W)
        
        # Decode
        Y = self.dec(z, skip_out)
        Y = Y.reshape(B, self.T_out, self.out_channels, H, W)
        return Y