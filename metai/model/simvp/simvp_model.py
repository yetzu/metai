import torch
from torch import nn

from .simvp_module import (
    BasicConv2d, 
    ConvSC,
    GroupConv2d, 
    gInception_ST, 
    AttentionModule, 
    SpatialAttention, 
    GASubBlock, 
    ConvMixerSubBlock, 
    ConvNeXtSubBlock, 
    HorNetSubBlock, 
    MLPMixerSubBlock, 
    MogaSubBlock, 
    PoolFormerSubBlock, 
    SwinSubBlock, 
    UniformerSubBlock, 
    VANSubBlock, 
    ViTSubBlock, 
    TemporalAttention, 
    TemporalAttentionModule, 
    TAUSubBlock,
    MambaSubBlock
)

class SimVP_Model(nn.Module):
    r"""SimVP Model (SOTA Optimized)
    
    主要改进：
    1. 支持输入输出变长 (In=10 -> Out=20)。
    2. 支持 Early Fusion 的多通道输入 (C=54)。
    3. 集成双向解耦 Mamba 模块。
    """

    def __init__(self, in_shape, hid_S=16, hid_T=256, N_S=4, N_T=4, model_type='gSTA',
                 mlp_ratio=8., drop=0.0, drop_path=0.0, spatio_kernel_enc=3,
                 spatio_kernel_dec=3, act_inplace=True, out_channels=None, 
                 aft_seq_length=None, **kwargs):
        super(SimVP_Model, self).__init__()
        
        # T: 输入帧数 (10), C: 输入通道数 (54)
        T, C, H, W = in_shape  
        
        # [关键 1] 确定输出帧数。如果未指定，默认与输入相同
        self.T_out = aft_seq_length if aft_seq_length is not None else T
        
        if out_channels is None:
            out_channels = C
        self.out_channels = out_channels
        
        # 下采样倍率
        H, W = int(H / 2**(N_S/2)), int(W / 2**(N_S/2))
        act_inplace = False
        
        # Encoder: 负责将 (54, H, W) 压缩为 (hid_S, H', W')
        # 它会自动适配 config 中的 in_shape 通道数
        self.enc = Encoder(C, hid_S, N_S, spatio_kernel_enc, act_inplace=act_inplace)
        
        # Decoder: 负责解码，输入输出通道都是 hid_S
        self.dec = Decoder(hid_S, hid_S, N_S, spatio_kernel_dec, act_inplace=act_inplace)

        model_type = 'gsta' if model_type is None else model_type.lower()
        
        # [关键 2] 计算 Translator 的输入输出通道
        # 输入: T_in * hid_S
        # 输出: T_out * hid_S (实现了时间维度的扩展)
        channel_in = T * hid_S
        channel_out = self.T_out * hid_S 

        if model_type == 'incepu':
            self.hid = MidIncepNet(channel_in, hid_T, N_T) # IncepU 暂未适配变长，建议用 Mamba
        else:
            self.hid = MidMetaNet(channel_in, hid_T, N_T,
                input_resolution=(H, W), model_type=model_type,
                mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path,
                channel_out=channel_out) # 传入目标输出通道数

        # Readout: 将特征图映射回预测目标 (out_channels=1)
        self.readout = nn.Conv2d(hid_S, out_channels, kernel_size=1)

    def forward(self, x_raw, **kwargs):
        # x_raw: [B, T_in, C_in, H, W]
        B, T, C, H, W = x_raw.shape
        x = x_raw.view(B*T, C, H, W)

        # 1. Encoder
        embed, skip = self.enc(x)
        _, C_, H_, W_ = embed.shape # C_ = hid_S

        # 2. Translator (MidMetaNet)
        z = embed.view(B, T, C_, H_, W_)
        # hid shape: [B, T_out, C_, H_, W_] (时间维度已从 10 变为 20)
        hid = self.hid(z)
        
        # 准备 Decoder 输入: [B*T_out, C_, H_, W_]
        hid = hid.reshape(B * self.T_out, C_, H_, W_)

        # [关键 3] Skip Connection 对齐
        # Encoder 输出了 T_in 帧的 Skip，但 Decoder 需要 T_out 帧
        # 策略：取 Encoder 最后一帧的 Skip 特征，复制 T_out 次
        # 这意味着 Decoder 在解码每一帧时，都使用最新的观测时刻的地形/背景特征
        skip = skip.view(B, T, C_, H_, W_)
        skip_last = skip[:, -1:, ...] # [B, 1, C_, H_, W_]
        
        # 扩展到 T_out 长度
        skip_out = skip_last.expand(-1, self.T_out, -1, -1, -1)
        skip_out = skip_out.reshape(B * self.T_out, C_, H_, W_)
        
        # 3. Decoder
        Y = self.dec(hid, skip_out)
        
        # 4. Readout
        Y = self.readout(Y)
        Y = Y.reshape(B, self.T_out, self.out_channels, H, W)
        return Y


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
              ConvSC(C_in, C_hid, spatio_kernel, downsampling=samplings[0],
                     act_inplace=act_inplace),
            *[ConvSC(C_hid, C_hid, spatio_kernel, downsampling=s,
                     act_inplace=act_inplace) for s in samplings[1:]]
        )

    def forward(self, x):
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
            *[ConvSC(C_hid, C_hid, spatio_kernel, upsampling=s,
                     act_inplace=act_inplace) for s in samplings[:-1]],
              ConvSC(C_hid, C_hid, spatio_kernel, upsampling=samplings[-1],
                     act_inplace=act_inplace)
        )
        self.readout = nn.Conv2d(C_hid, C_out, 1)

    def forward(self, hid, enc1=None):
        for i in range(0, len(self.dec)-1):
            hid = self.dec[i](hid)
        Y = self.dec[-1](hid + enc1)
        Y = self.readout(Y)
        return Y


class MidMetaNet(nn.Module):
    """The hidden Translator of MetaFormer for SimVP"""

    def __init__(self, channel_in, channel_hid, N2,
                 input_resolution=None, model_type=None,
                 mlp_ratio=4., drop=0.0, drop_path=0.1, 
                 channel_out=None): # [新增] 接收 channel_out
        super(MidMetaNet, self).__init__()
        assert N2 >= 2 and mlp_ratio > 1
        self.N2 = N2
        
        # 如果未指定输出通道，默认与输入相同（兼容旧逻辑）
        if channel_out is None:
            channel_out = channel_in
        self.channel_out = channel_out
            
        dpr = [x.item() for x in torch.linspace(1e-2, drop_path, self.N2)]

        # 1. Input Layer: channel_in -> channel_hid
        enc_layers = [MetaBlock(
            channel_in, channel_hid, input_resolution, model_type,
            mlp_ratio, drop, drop_path=dpr[0], layer_i=0)]
        
        # 2. Middle Layers: channel_hid -> channel_hid
        for i in range(1, N2-1):
            enc_layers.append(MetaBlock(
                channel_hid, channel_hid, input_resolution, model_type,
                mlp_ratio, drop, drop_path=dpr[i], layer_i=i))
                
        # 3. Output Layer: channel_hid -> channel_out (实现时间维度扩展)
        enc_layers.append(MetaBlock(
            channel_hid, channel_out, input_resolution, model_type,
            mlp_ratio, drop, drop_path=drop_path, layer_i=N2-1))
            
        self.enc = nn.Sequential(*enc_layers)

    def forward(self, x):
        # x: [B, T_in, C, H, W]
        B, T, C, H, W = x.shape
        # 将时间维度并入通道维度: [B, T*C, H, W]
        x = x.reshape(B, T*C, H, W)

        z = x
        for i in range(self.N2):
            z = self.enc[i](z)

        # z: [B, channel_out, H, W]
        # 还原时间维度: [B, T_out, C, H, W]
        # 因为 channel_out = T_out * C, 所以 T_out = channel_out // C
        T_out = self.channel_out // C 
        y = z.reshape(B, T_out, C, H, W)
        return y


class MetaBlock(nn.Module):
    """
    SimVP 的标准 MetaBlock。
    注意：这里的 MambaSubBlock 已经假设你在 simvp_module.py 中使用了最新的双向解耦版本。
    """
    def __init__(self, in_channels, out_channels, input_resolution=None, model_type=None,
                 mlp_ratio=8., drop=0.0, drop_path=0.0, layer_i=0):
        super(MetaBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        model_type = model_type.lower() if model_type is not None else 'gsta'

        if model_type == 'gsta':
            self.block = GASubBlock(
                in_channels, kernel_size=21, mlp_ratio=mlp_ratio,
                drop=drop, drop_path=drop_path, act_layer=nn.GELU)
        # ... (省略其他类型的 elif，保持原样即可)
        elif model_type == 'tau':
            self.block = TAUSubBlock(
                in_channels, kernel_size=21, mlp_ratio=mlp_ratio,
                drop=drop, drop_path=drop_path, act_layer=nn.GELU)
        elif model_type == 'mamba':
            # 调用 simvp_module.py 中的 MambaSubBlock
            self.block = MambaSubBlock(
                in_channels, mlp_ratio=mlp_ratio, 
                drop=drop, drop_path=drop_path, act_layer=nn.GELU)
        else:
            # 默认回退
            self.block = GASubBlock(
                in_channels, kernel_size=21, mlp_ratio=mlp_ratio,
                drop=drop, drop_path=drop_path, act_layer=nn.GELU)

        # 如果输入输出通道不一致（发生在第一层和最后一层），使用 1x1 卷积进行投影
        if in_channels != out_channels:
            self.reduction = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        z = self.block(x)
        return z if self.in_channels == self.out_channels else self.reduction(z)

class MidIncepNet(nn.Module):
    """The hidden Translator of IncepNet for SimVPv1"""

    def __init__(self, channel_in, channel_hid, N2, incep_ker=[3,5,7,11], groups=8, **kwargs):
        super(MidIncepNet, self).__init__()
        assert N2 >= 2 and len(incep_ker) > 1
        self.N2 = N2
        enc_layers = [gInception_ST(
            channel_in, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups)]
        for i in range(1,N2-1):
            enc_layers.append(
                gInception_ST(channel_hid, channel_hid//2, channel_hid,
                              incep_ker=incep_ker, groups=groups))
        enc_layers.append(
                gInception_ST(channel_hid, channel_hid//2, channel_hid,
                              incep_ker=incep_ker, groups=groups))
        dec_layers = [
                gInception_ST(channel_hid, channel_hid//2, channel_hid,
                              incep_ker=incep_ker, groups=groups)]
        for i in range(1,N2-1):
            dec_layers.append(
                gInception_ST(2*channel_hid, channel_hid//2, channel_hid,
                              incep_ker=incep_ker, groups=groups))
        dec_layers.append(
                gInception_ST(2*channel_hid, channel_hid//2, channel_in,
                              incep_ker=incep_ker, groups=groups))

        self.enc = nn.Sequential(*enc_layers)
        self.dec = nn.Sequential(*dec_layers)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T*C, H, W)

        # encoder
        skips = []
        z = x
        for i in range(self.N2):
            z = self.enc[i](z)
            if i < self.N2-1:
                skips.append(z)
        # decoder
        z = self.dec[0](z)
        for i in range(1,self.N2):
            z = self.dec[i](torch.cat([z, skips[-i]], dim=1) )

        y = z.reshape(B, T, C, H, W)
        return y


