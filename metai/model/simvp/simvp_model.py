import torch
from torch import nn

from .simvp_module import BasicConv2d, ConvSC, GroupConv2d, gInception_ST, AttentionModule, SpatialAttention, GASubBlock, ConvMixerSubBlock, ConvNeXtSubBlock, HorNetSubBlock, MLPMixerSubBlock, MogaSubBlock, PoolFormerSubBlock, SwinSubBlock, UniformerSubBlock, VANSubBlock, ViTSubBlock, TemporalAttention, TemporalAttentionModule, TAUSubBlock

class SimVP_Model(nn.Module):
    r"""SimVP Model

    Implementation of `SimVP: Simpler yet Better Video Prediction
    <https://arxiv.org/abs/2206.05099>`_.

    """

    def __init__(self, in_shape, hid_S=16, hid_T=256, N_S=4, N_T=4, model_type='gSTA',
                 mlp_ratio=8., drop=0.0, drop_path=0.0, spatio_kernel_enc=3,
                 spatio_kernel_dec=3, act_inplace=True, out_channels=None, **kwargs):
        super(SimVP_Model, self).__init__()
        T, C, H, W = in_shape  # T is pre_seq_length
        # 如果未指定 out_channels，则使用输入通道数 C（保持向后兼容）
        if out_channels is None:
            out_channels = C
        self.out_channels = out_channels
        H, W = int(H / 2**(N_S/2)), int(W / 2**(N_S/2))  # downsample 1 / 2**(N_S/2)
        act_inplace = False
        self.enc = Encoder(C, hid_S, N_S, spatio_kernel_enc, act_inplace=act_inplace)
        # Decoder 只负责上采样，输出 hid_S 通道（不进行通道映射）
        # 通道映射由 SimVP_Model 的 readout 层完成，避免浪费计算量
        self.dec = Decoder(hid_S, hid_S, N_S, spatio_kernel_dec, act_inplace=act_inplace)

        model_type = 'gsta' if model_type is None else model_type.lower()
        if model_type == 'incepu':
            self.hid = MidIncepNet(T*hid_S, hid_T, N_T)
        else:
            self.hid = MidMetaNet(T*hid_S, hid_T, N_T,
                input_resolution=(H, W), model_type=model_type,
                mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)
        # 将隐藏层映射为 out_channels 通道（对于降水预测，out_channels=1）
        # 这样避免了 Decoder 输出 28 通道后再切片，节省计算量并避免无效梯度干扰
        self.readout = nn.Conv2d(hid_S, out_channels, kernel_size=1)

    def forward(self, x_raw, **kwargs):
        B, T, C, H, W = x_raw.shape
        x = x_raw.view(B*T, C, H, W)

        embed, skip = self.enc(x)
        _, C_, H_, W_ = embed.shape

        z = embed.view(B, T, C_, H_, W_)
        hid = self.hid(z)
        hid = hid.reshape(B*T, C_, H_, W_)

        # Decoder 输出 hid_S 通道的特征图
        Y = self.dec(hid, skip)
        # 使用 readout 层将 hid_S 通道映射到 out_channels（对于降水预测，out_channels=1）
        # 这样避免了预测 28 通道后再切片，节省计算量并避免无效梯度干扰
        Y = self.readout(Y)
        Y = Y.reshape(B, T, self.out_channels, H, W)
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

    def forward(self, x):  # B*4, 3, 128, 128
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
        # 移除ReLU，允许输出负数，避免梯度消失问题
        # 数据归一化到[0,1]，但模型输出可以是负数，在损失函数和后处理中处理
        self.readout = nn.Conv2d(C_hid, C_out, 1)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        """兼容旧checkpoint：如果readout是Sequential，则从readout.0加载权重"""
        # 检查是否有旧的checkpoint格式（readout.0.weight）
        old_weight_key = prefix + 'readout.0.weight'
        old_bias_key = prefix + 'readout.0.bias'
        new_weight_key = prefix + 'readout.weight'
        new_bias_key = prefix + 'readout.bias'
        
        if old_weight_key in state_dict and new_weight_key not in state_dict:
            # 旧格式存在，新格式不存在，需要转换
            state_dict[new_weight_key] = state_dict.pop(old_weight_key)
            if old_bias_key in state_dict:
                state_dict[new_bias_key] = state_dict.pop(old_bias_key)
        
        # 调用父类方法
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def forward(self, hid, enc1=None):
        for i in range(0, len(self.dec)-1):
            hid = self.dec[i](hid)
        Y = self.dec[-1](hid + enc1)
        Y = self.readout(Y)
        return Y


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


class MetaBlock(nn.Module):
    """The hidden Translator of MetaFormer for SimVP"""

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
        elif model_type == 'convmixer':
            self.block = ConvMixerSubBlock(in_channels, kernel_size=11, activation=nn.GELU)
        elif model_type == 'convnext':
            self.block = ConvNeXtSubBlock(
                in_channels, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)
        elif model_type == 'hornet':
            self.block = HorNetSubBlock(in_channels, mlp_ratio=mlp_ratio, drop_path=drop_path)
        elif model_type in ['mlp', 'mlpmixer']:
            self.block = MLPMixerSubBlock(
                in_channels, input_resolution, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)
        elif model_type in ['moga', 'moganet']:
            self.block = MogaSubBlock(
                in_channels, mlp_ratio=mlp_ratio, drop_rate=drop, drop_path_rate=drop_path)
        elif model_type == 'poolformer':
            self.block = PoolFormerSubBlock(
                in_channels, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)
        elif model_type == 'swin':
            self.block = SwinSubBlock(
                in_channels, input_resolution, layer_i=layer_i, mlp_ratio=mlp_ratio,
                drop=drop, drop_path=drop_path)
        elif model_type == 'uniformer':
            block_type = 'MHSA' if in_channels == out_channels and layer_i > 0 else 'Conv'
            self.block = UniformerSubBlock(
                in_channels, mlp_ratio=mlp_ratio, drop=drop,
                drop_path=drop_path, block_type=block_type)
        elif model_type == 'van':
            self.block = VANSubBlock(
                in_channels, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path, act_layer=nn.GELU)
        elif model_type == 'vit':
            self.block = ViTSubBlock(
                in_channels, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)
        elif model_type == 'tau':
            self.block = TAUSubBlock(
                in_channels, kernel_size=21, mlp_ratio=mlp_ratio,
                drop=drop, drop_path=drop_path, act_layer=nn.GELU)
        else:
            assert False and "Invalid model_type in SimVP"

        if in_channels != out_channels:
            self.reduction = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        z = self.block(x)
        return z if self.in_channels == self.out_channels else self.reduction(z)


class MidMetaNet(nn.Module):
    """The hidden Translator of MetaFormer for SimVP"""

    def __init__(self, channel_in, channel_hid, N2,
                 input_resolution=None, model_type=None,
                 mlp_ratio=4., drop=0.0, drop_path=0.1):
        super(MidMetaNet, self).__init__()
        assert N2 >= 2 and mlp_ratio > 1
        self.N2 = N2
        dpr = [  # stochastic depth decay rule
            x.item() for x in torch.linspace(1e-2, drop_path, self.N2)]

        # downsample
        enc_layers = [MetaBlock(
            channel_in, channel_hid, input_resolution, model_type,
            mlp_ratio, drop, drop_path=dpr[0], layer_i=0)]
        # middle layers
        for i in range(1, N2-1):
            enc_layers.append(MetaBlock(
                channel_hid, channel_hid, input_resolution, model_type,
                mlp_ratio, drop, drop_path=dpr[i], layer_i=i))
        # upsample
        enc_layers.append(MetaBlock(
            channel_hid, channel_in, input_resolution, model_type,
            mlp_ratio, drop, drop_path=drop_path, layer_i=N2-1))
        self.enc = nn.Sequential(*enc_layers)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T*C, H, W)

        z = x
        for i in range(self.N2):
            z = self.enc[i](z)

        y = z.reshape(B, T, C, H, W)
        return y
