import torch
import torch.nn as nn
from .modules import ConvSC, STMambaBlock, TimeAlignBlock

def sampling_generator(N, reverse=False):
    samplings = [False, True] * (N // 2)
    if reverse: return list(reversed(samplings[:N]))
    else: return samplings[:N]

class Encoder(nn.Module):
    """
    3D-2D Hybrid Encoder
    先通过 3D Stem 提取短时空特征，再通过 2D ConvSC 提取空间层级特征。
    """
    def __init__(self, C_in, C_hid, N_S, spatio_kernel):
        super().__init__()
        
        # [New] 3D Stem Layer: 提取短时空特征 (C_in -> C_hid)
        # kernel_size=(3, 3, 3) 用于同时捕捉时空特征，padding=(1, 1, 1) 保持尺寸不变
        self.stem = nn.Sequential(
            nn.Conv3d(C_in, C_hid, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.GroupNorm(2, C_hid), # GroupNorm 对小 Batch Size 更友好
            nn.SiLU(inplace=True)
        )
        
        samplings = sampling_generator(N_S)
        self.enc = nn.Sequential(
            # [修改] 第一层 ConvSC 的输入通道变为 C_hid (因为 Stem 已经将通道数映射到了 C_hid)
            ConvSC(C_hid, C_hid, spatio_kernel, downsampling=samplings[0]),
            *[ConvSC(C_hid, C_hid, spatio_kernel, downsampling=s) for s in samplings[1:]]
        )

    def forward(self, x):
        # [修改] 输入 x 此时为 5D 张量: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        
        # 1. 3D Stem 处理
        # Conv3d 需要输入格式为 [B, C, T, H, W]
        x = x.permute(0, 2, 1, 3, 4)
        x = self.stem(x) 
        # Output: [B, C_hid, T, H, W]
        
        # 2. 转换回 2D Encoder 需要的格式
        # Permute back to [B, T, C_hid, H, W] -> Flatten to [B*T, C_hid, H, W]
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(B * T, -1, H, W)
        
        # 3. 后续 2D ConvSC 处理保持不变
        enc1 = self.enc[0](x)
        latent = enc1
        for i in range(1, len(self.enc)):
            latent = self.enc[i](latent)
        return latent, enc1

class Decoder(nn.Module):
    """3D Decoder using 2D Convolutions"""
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
        # Skip connection fusion
        Y = self.dec[-1](hid + skip)
        return self.readout(Y)

class EvolutionNet(nn.Module):
    """
    Mamba-based Evolution Network
    Uses STMambaBlocks for spatiotemporal modeling in latent space.
    """
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
        # Permute to Channel-last for Linear projection: [B, T, H, W, C]
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = self.proj_in(x)
        
        # Permute for STMambaBlock: [B, T, C_hid, H, W] (assuming STMamba handles this layout)
        # Wait, modules.py STMambaBlock expects [B, T, C, H, W] or similar token handling.
        # Let's standardize to [B, T, C, H, W] for blocks
        x = x.permute(0, 1, 4, 2, 3).contiguous()
        
        for layer in self.layers:
            x = layer(x)
            
        # Project back
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
        
        # Renamed FeatureEncoder -> Encoder
        self.enc = Encoder(C, hid_S, N_S, spatio_kernel_enc)
        
        self.evolution = EvolutionNet(hid_S, hid_T, N_T, drop=0.0, drop_path=0.1)
        
        # Latent temporal extrapolation (T_in -> T_out)
        self.latent_time_proj = nn.Linear(T_in, self.T_out)
        
        # Renamed FeatureDecoder -> Decoder
        self.dec = Decoder(hid_S, self.out_channels, N_S, spatio_kernel_dec)
        
        # Skip connection projector
        self.skip_proj = TimeAlignBlock(T_in, self.T_out, hid_S)
        
        self._init_time_proj()

    def _init_time_proj(self):
        nn.init.normal_(self.latent_time_proj.weight, mean=0.0, std=0.01)
        with torch.no_grad():
            if self.latent_time_proj.weight.shape[1] > 0:
                # Initialize to copy the last frame
                self.latent_time_proj.weight[:, -1].fill_(1.0)
        nn.init.constant_(self.latent_time_proj.bias, 0)

    def forward(self, x_raw):
        # x_raw: [B, T_in, C_in, H, W]
        B, T_in, C_in, H, W = x_raw.shape
        
        # [修改] 直接传入 5D 数据，Encoder 内部处理 3D Stem 和展平
        embed, skip = self.enc(x_raw) 
        
        # embed: [B*T_in, C_hid, H', W']
        # skip:  [B*T_in, C_hid, H, W] (第一层特征)
        
        _, C_hid, H_, W_ = embed.shape 
        
        # 1. Evolution in Latent Space
        # Restore 5D for EvolutionNet: [B, T_in, C_hid, H', W']
        z = embed.view(B, T_in, C_hid, H_, W_)
        z = self.evolution(z)
        
        # 2. Temporal Extrapolation (T_in -> T_out)
        # Permute to [B, C, H, W, T] for Linear projection on T axis
        z = z.permute(0, 2, 3, 4, 1) 
        z = self.latent_time_proj(z) 
        z = z.permute(0, 4, 1, 2, 3).contiguous() # [B, T_out, C_hid, H', W']
        
        # Flatten for Decoder: [B*T_out, C_hid, H', W']
        z = z.view(B * self.T_out, C_hid, H_, W_)
        
        # 3. Skip Connection Processing
        skip = skip.view(B, T_in, C_hid, H, W)
        skip_out = self.skip_proj(skip) # [B, T_out, C_hid, H, W]
        skip_out = skip_out.view(B * self.T_out, C_hid, H, W)
        
        # 4. Decoding
        Y = self.dec(z, skip_out)
        
        # 5. Reshape to Output Sequence
        Y = Y.reshape(B, self.T_out, self.out_channels, H, W)
        return Y