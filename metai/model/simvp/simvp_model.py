import torch
from torch import nn
from .simvp_module import (
    ConvSC, 
    GASubBlock, 
    # 引入新模块
    VideoMambaSubBlock, 
    TimeAwareSkipBlock
)

def sampling_generator(N, reverse=False):
    samplings = [False, True] * (N // 2)
    if reverse: return list(reversed(samplings[:N]))
    else: return samplings[:N]

class Encoder(nn.Module):
    """3D Encoder for SimVP"""
    def __init__(self, C_in, C_hid, N_S, spatio_kernel, act_inplace=True):
        samplings = sampling_generator(N_S)
        super(Encoder, self).__init__()
        self.enc = nn.Sequential(
              ConvSC(C_in, C_hid, spatio_kernel, downsampling=samplings[0], act_inplace=act_inplace),
            *[ConvSC(C_hid, C_hid, spatio_kernel, downsampling=s, act_inplace=act_inplace) for s in samplings[1:]]
        )

    def forward(self, x):
        # x: [B*T, C, H, W]
        enc1 = self.enc[0](x)
        latent = enc1
        for i in range(1, len(self.enc)):
            latent = self.enc[i](latent)
        return latent, enc1

class Decoder(nn.Module):
    """3D Decoder for SimVP"""
    def __init__(self, C_hid, C_out, N_S, spatio_kernel, act_inplace=True):
        samplings = sampling_generator(N_S, reverse=True)
        super(Decoder, self).__init__()
        self.dec = nn.Sequential(
            *[ConvSC(C_hid, C_hid, spatio_kernel, upsampling=s, act_inplace=act_inplace) for s in samplings[:-1]],
              ConvSC(C_hid, C_hid, spatio_kernel, upsampling=samplings[-1], act_inplace=act_inplace)
        )
        self.readout = nn.Conv2d(C_hid, C_out, 1)

    def forward(self, hid, enc1=None):
        for i in range(0, len(self.dec)-1):
            hid = self.dec[i](hid)
        # Skip connection fusion
        Y = self.dec[-1](hid + enc1)
        Y = self.readout(Y)
        return Y

class MetaBlock(nn.Module):
    """SimVP MetaBlock with support for VideoMamba"""
    def __init__(self, in_channels, out_channels, input_resolution=None, model_type=None,
                 mlp_ratio=8., drop=0.0, drop_path=0.0, layer_i=0):
        super(MetaBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        model_type = model_type.lower() if model_type is not None else 'gsta'

        if model_type == 'video_mamba':
            # [NEW] VideoMamba
            self.block = VideoMambaSubBlock(
                in_channels, mlp_ratio=mlp_ratio, 
                drop=drop, drop_path=drop_path, act_layer=nn.GELU
            )
            # 如果通道数不同，VideoMamba 内部通常不处理维度变化，这里假设维度一致
            # 或者可以在 block 后加 Linear，这里简化处理
            self.reduction = nn.Identity()
        elif model_type == 'gsta':
            self.block = GASubBlock(
                in_channels, kernel_size=21, mlp_ratio=mlp_ratio,
                drop=drop, drop_path=drop_path, act_layer=nn.GELU)
            self.reduction = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        else:
            # Default fallback
            self.block = GASubBlock(in_channels, kernel_size=21, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)
            self.reduction = nn.Identity()

    def forward(self, x):
        z = self.block(x)
        return self.reduction(z)

class MidMetaNet(nn.Module):
    """The hidden Translator of MetaFormer for SimVP"""
    def __init__(self, channel_in, channel_hid, N2,
                 input_resolution=None, model_type=None,
                 mlp_ratio=4., drop=0.0, drop_path=0.1, 
                 channel_out=None):
        super(MidMetaNet, self).__init__()
        assert N2 >= 2 and mlp_ratio > 1
        self.model_type = model_type.lower() if model_type else 'gsta'
        
        # [NEW] VideoMamba 逻辑: 处理 5D 张量，不进行 T-Channel 混合
        if self.model_type == 'video_mamba':
            dpr = [x.item() for x in torch.linspace(1e-2, drop_path, N2)]
            enc_layers = []
            
            # 维度适配: channel_in (hid_S) -> channel_hid (hid_T)
            # 这里的 Linear 作用于 Channel 维度
            self.proj_in = nn.Linear(channel_in, channel_hid)
            self.proj_out = nn.Linear(channel_hid, channel_in) # 假设输出维度还原回 hid_S
            
            for i in range(N2):
                enc_layers.append(MetaBlock(
                    channel_hid, channel_hid, input_resolution, 
                    model_type='video_mamba', 
                    mlp_ratio=mlp_ratio, drop=drop, drop_path=dpr[i]
                ))
            self.enc = nn.Sequential(*enc_layers)
            
        # 标准 SimVP 逻辑: 处理 4D 张量 (Time folded into Channel)
        else:
            if channel_out is None: channel_out = channel_in
            dpr = [x.item() for x in torch.linspace(1e-2, drop_path, N2)]
            enc_layers = [MetaBlock(channel_in, channel_hid, input_resolution, model_type, mlp_ratio, drop, dpr[0], 0)]
            for i in range(1, N2-1):
                enc_layers.append(MetaBlock(channel_hid, channel_hid, input_resolution, model_type, mlp_ratio, drop, dpr[i], i))
            enc_layers.append(MetaBlock(channel_hid, channel_out, input_resolution, model_type, mlp_ratio, drop, drop_path, N2-1))
            self.enc = nn.Sequential(*enc_layers)

    def forward(self, x):
        # VideoMamba: x is [B, T, C, H, W]
        if self.model_type == 'video_mamba':
            # 1. Project Channels: [B, T, C_in, H, W] -> [B, T, H, W, C_in] -> [B, T, H, W, C_hid]
            x = x.permute(0, 1, 3, 4, 2).contiguous()
            x = self.proj_in(x)
            
            # 2. Restore to [B, T, C_hid, H, W] for Block
            x = x.permute(0, 1, 4, 2, 3).contiguous()
            
            # 3. Through Blocks
            for layer in self.enc:
                x = layer(x)
            
            # 4. Project Back: [B, T, C_hid, H, W] -> [B, T, C_in, H, W]
            x = x.permute(0, 1, 3, 4, 2).contiguous()
            x = self.proj_out(x)
            x = x.permute(0, 1, 4, 2, 3).contiguous()
            return x
            
        # Standard: x is [B, T_in, C, H, W] -> [B, T_in*C, H, W]
        else:
            B, T, C, H, W = x.shape
            x = x.reshape(B, T*C, H, W)
            z = x
            for i in range(len(self.enc)):
                z = self.enc[i](z)
            # Output: [B, T_out*C, H, W] -> [B, T_out, C, H, W]
            # 注意：Standard 模式下 channel_out 已经隐含了 T_out
            T_out = z.shape[1] // C
            y = z.reshape(B, T_out, C, H, W)
            return y

class SimVP_Model(nn.Module):
    """SimVP Model with VideoMamba and Time-Aware Skip Connections"""
    def __init__(self, in_shape, hid_S=16, hid_T=256, N_S=4, N_T=4, model_type='gSTA',
                 mlp_ratio=8., drop=0.0, drop_path=0.0, spatio_kernel_enc=3,
                 spatio_kernel_dec=3, act_inplace=True, out_channels=None, 
                 aft_seq_length=None, **kwargs):
        super(SimVP_Model, self).__init__()
        
        T_in, C, H, W = in_shape
        self.T_in = T_in
        self.T_out = aft_seq_length if aft_seq_length is not None else T_in
        if out_channels is None: out_channels = C
        self.out_channels = out_channels
        
        # Calculate latent resolution
        H_, W_ = int(H / 2**(N_S/2)), int(W / 2**(N_S/2))
        act_inplace = False
        
        # 1. Encoder
        self.enc = Encoder(C, hid_S, N_S, spatio_kernel_enc, act_inplace=act_inplace)
        
        # 2. Translator
        model_type = model_type.lower() if model_type else 'gsta'
        self.model_type = model_type
        
        if model_type == 'video_mamba':
            # VideoMamba Translator
            self.hid = MidMetaNet(
                channel_in=hid_S, # 单帧通道
                channel_hid=hid_T, 
                N2=N_T,
                input_resolution=(H_, W_), 
                model_type='video_mamba',
                mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path
            )
            # [NEW] Latent Space Temporal Extrapolation: T_in -> T_out
            # 使用 Linear 在 T 轴上投影
            self.latent_time_proj = nn.Linear(T_in, self.T_out)
        else:
            # Standard SimVP Translator
            channel_in = T_in * hid_S
            channel_out = self.T_out * hid_S
            self.hid = MidMetaNet(channel_in, hid_T, N_T, (H_, W_), model_type, mlp_ratio, drop, drop_path, channel_out)
            self.latent_time_proj = None

        # 3. Decoder
        self.dec = Decoder(hid_S, hid_S, N_S, spatio_kernel_dec, act_inplace=act_inplace)

        # 4. [NEW] Time-Aware Skip Connection
        # 将 Encoder 的 Skip 特征 (B, T_in, C, H, W) 映射到 (B, T_out, C, H, W)
        self.skip_connect_proj = TimeAwareSkipBlock(T_in, self.T_out, hid_S)

        # 5. Readout
        self.readout = nn.Conv2d(hid_S, out_channels, kernel_size=1)
        nn.init.constant_(self.readout.bias, -5.0)

        # [NEW] 针对时序投影层的特殊初始化
        if self.latent_time_proj is not None:
            # 初始化为：未来的特征 ≈ 过去最后一帧的特征
            nn.init.normal_(self.latent_time_proj.weight, mean=0.0, std=0.01)
            with torch.no_grad():
                self.latent_time_proj.weight[:, -1].fill_(1.0)
            nn.init.constant_(self.latent_time_proj.bias, 0)

    def forward(self, x_raw, **kwargs):
        # x_raw: [B, T_in, C_in, H, W]
        B, T_in, C_in, H, W = x_raw.shape
        x = x_raw.view(B*T_in, C_in, H, W)

        # 1. Encoder
        # embed: [B*T_in, hid_S, H', W']
        # skip:  [B*T_in, hid_S, H, W] (取第一层特征)
        embed, skip = self.enc(x)
        _, C_hid, H_, W_ = embed.shape 

        # 2. Translator
        if self.model_type == 'video_mamba':
            # === VideoMamba Path ===
            # Restore 5D: [B, T_in, C_hid, H', W']
            z = embed.view(B, T_in, C_hid, H_, W_)
            
            # (A) VideoMamba Feature Extraction (Keep T_in)
            hid = self.hid(z) # [B, T_in, C_hid, H', W']
            
            # (B) Latent Temporal Extrapolation (T_in -> T_out)
            # Permute to [B, C, H, W, T] for Linear
            hid = hid.permute(0, 2, 3, 4, 1) 
            hid = self.latent_time_proj(hid) # [B, C, H, W, T_out]
            hid = hid.permute(0, 4, 1, 2, 3).contiguous() # [B, T_out, C, H, W]
            
            # Flatten for Decoder: [B*T_out, C_hid, H', W']
            hid = hid.view(B * self.T_out, C_hid, H_, W_)
            
        else:
            # === Standard SimVP Path ===
            z = embed.view(B, T_in, C_hid, H_, W_)
            hid = self.hid(z) # Returns [B, T_out, C_hid, H', W'] or flattened
            hid = hid.reshape(B * self.T_out, C_hid, H_, W_)

        # 3. Time-Aware Skip Connection
        # skip: [B*T_in, C_skip, H, W]
        _, C_skip, H_skip, W_skip = skip.shape
        skip = skip.view(B, T_in, C_skip, H_skip, W_skip)
        
        # Apply projection: T_in -> T_out
        skip_out = self.skip_connect_proj(skip) # [B, T_out, C_skip, H, W]
        skip_out = skip_out.view(B * self.T_out, C_skip, H_skip, W_skip)
        
        # 4. Decoder
        Y = self.dec(hid, skip_out)
        
        # 5. Readout
        Y = self.readout(Y)
        Y = Y.reshape(B, self.T_out, self.out_channels, H, W)
        return Y