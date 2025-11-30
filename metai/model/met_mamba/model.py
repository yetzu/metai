# metai/model/met_mamba/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers.weight_init import trunc_normal_
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
    def __init__(self, C_hid, C_out, N_S, spatio_kernel):
        super().__init__()
        samplings = sampling_generator(N_S, reverse=True)
        layers = []
        for i, s in enumerate(samplings):
            # Strategy: Use ResizeConv for last 2 layers to reduce artifacts
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

class EvolutionNet(nn.Module):
    def __init__(self, dim_in, dim_hid, num_layers, drop=0., drop_path=0., 
                 mamba_kwargs={}, use_checkpoint=True, max_t=64, input_resolution=(64, 64)):
        super().__init__()
        self.proj_in = nn.Linear(dim_in, dim_hid)
        self.proj_out = nn.Linear(dim_hid, dim_in)
        
        # 3D Absolute Positional Embeddings
        self.pos_embed_t = nn.Parameter(torch.zeros(1, max_t, dim_hid, 1, 1)) 
        
        # Dynamic Spatial PE init
        H_feat, W_feat = input_resolution
        self.pos_embed_s = nn.Parameter(torch.zeros(1, 1, dim_hid, H_feat, W_feat)) 
        
        # Time Prompt (Optimization for sparse radar data)
        self.time_prompt = nn.Parameter(torch.zeros(1, max_t, dim_hid, 1, 1))

        trunc_normal_(self.pos_embed_t, std=0.02)
        trunc_normal_(self.pos_embed_s, std=0.02)
        trunc_normal_(self.time_prompt, std=0.02)
        
        dpr = [x.item() for x in torch.linspace(1e-2, drop_path, num_layers)]
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
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        
        x = x.permute(0, 1, 3, 4, 2).contiguous() # [B, T, H, W, C]
        x = self.proj_in(x)
        x = x.permute(0, 1, 4, 2, 3).contiguous() # [B, T, C_hid, H, W]
        
        # --- Interpolation Logic ---
        # 1. Time Prompt & Time PE
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

        # 2. Spatial PE
        if (H, W) != self.pos_embed_s.shape[-2:]:
            s_pos = F.interpolate(
                self.pos_embed_s, size=(H, W), mode='bilinear', align_corners=False
            )
        else:
            s_pos = self.pos_embed_s

        # 3. Additive Injection
        x = x + t_prompt + t_pos + s_pos
        
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
        
        self.enc = Encoder(C, hid_S, N_S, spatio_kernel_enc)
        
        # Calculate resolution at evolution stage (downsampled N_S/2 times)
        # Assumes N_S=4 implies 2 downsamples usually, depends on sampling_generator logic
        # Here assuming 4 layers with [False, True, False, True] -> 4x downsample total
        ds_factor = 2 ** (N_S // 2) 
        evo_res = (H // ds_factor, W // ds_factor)

        self.evolution = EvolutionNet(
            hid_S, hid_T, N_T, 
            drop=0.0, drop_path=0.1, 
            mamba_kwargs=mamba_kwargs,
            use_checkpoint=use_checkpoint,
            input_resolution=evo_res
        )
        
        # [CRITICAL FIX] Latent Time Projection
        # Changed kernel_size to 1 to prevent mixing spatially distant flattened pixels.
        # Logic: We want to mix T_in -> T_out PER SPATIAL LOCATION.
        self.latent_time_proj = nn.Sequential(
            nn.Conv1d(T_in, self.T_out, kernel_size=1),
            nn.Conv1d(self.T_out, self.T_out, kernel_size=1), # Was kernel_size=3
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
        
        # 1. Encoder -> [B*T_in, C_hid, H_, W_]
        embed, skip = self.enc(x_raw) 
        _, C_hid, H_, W_ = embed.shape 
        
        # 2. Evolution -> [B, T_in, C_hid, H_, W_]
        z = embed.view(B, T_in, C_hid, H_, W_)
        z = self.evolution(z)
        
        # 3. Time Projection -> [B, T_out, C_hid, H_, W_]
        # Reshape to [B, T_in, Features] where Features = C*H*W
        # Conv1d operates on Dim 1 (T_in), so Kernel slides over Features (Dim 2)
        # Using kernel=1 ensures we independently mix time for each feature (pixel).
        z = z.reshape(B, T_in, -1) 
        z = self.latent_time_proj(z) 
        z = z.reshape(B * self.T_out, C_hid, H_, W_)
        
        # 4. Skip Connection
        skip = skip.view(B, T_in, C_hid, H, W)
        skip_out = self.skip_proj(skip) 
        skip_out = skip_out.view(B * self.T_out, C_hid, H, W)
        
        # 5. Decoder
        Y_diff = self.dec(z, skip_out)
        Y_diff = Y_diff.reshape(B, self.T_out, self.out_channels, H, W)
        
        # 6. Residual Connection (Prediction = Last Frame + Diff)
        last_frame = x_raw[:, -1:, :self.out_channels, :, :].detach() 
        Y = last_frame + Y_diff
        
        return Y