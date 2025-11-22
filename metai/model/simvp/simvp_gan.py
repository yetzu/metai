# metai/model/simvp/simvp_gan.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as l
from metai.model.simvp import SimVP

# ===========================
# 1. 判别器 (Video Discriminator) - 升级版
# ===========================
class VideoDiscriminator(nn.Module):
    """
    3D 判别器 (Video Discriminator)，用于判断视频序列的时空连贯性。
    输入: (B, C, T, H, W) -> 输出: (B, 1, T', H', W')
    用于解决“长时间序列衰弱”问题，判别器能感知时间维度的变化。
    """
    def __init__(self, in_channels=1):
        super().__init__()
        
        def disc_block(in_f, out_f, kernel_size=(3,4,4), stride=(1,2,2), padding=(1,1,1), bn=True):
            # 使用 Conv3d 处理时空特征 (Time, Height, Width)
            # kernel_size=(3,4,4) 表示在时间维度卷积核为3，空间维度为4
            layers = [nn.utils.spectral_norm(nn.Conv3d(in_f, out_f, kernel_size, stride, padding, bias=False))]
            if bn:
                layers.append(nn.InstanceNorm3d(out_f)) # GAN 推荐用 InstanceNorm
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.ModuleList([
            # Input: (B, 1, 20, 256, 256)
            nn.Sequential(*disc_block(in_channels, 32, bn=False)),
            # (B, 32, 20, 128, 128)
            nn.Sequential(*disc_block(32, 64)),
            # (B, 64, 20, 64, 64)
            nn.Sequential(*disc_block(64, 128)),
            # (B, 128, 20, 32, 32)
            nn.Sequential(*disc_block(128, 256)),
            # (B, 256, 20, 16, 16)
        ])
        
        self.final_conv = nn.Conv3d(256, 1, kernel_size=3, stride=1, padding=1) # Output Logits

    def forward(self, x, return_feats=False):
        # 支持 Feature Matching Loss：返回中间层特征
        # x shape: (B, C, T, H, W)
        feats = []
        out = x
        for layer in self.model:
            out = layer(out)
            if return_feats:
                feats.append(out)
        
        out = self.final_conv(out)
        
        if return_feats:
            return out, feats
        return out

# ===========================
# 2. UNet Refiner (Sequence-Aware) - 升级版
# ===========================
class SequenceRefiner(nn.Module):
    """
    时序感知 Refiner：基于 2D UNet，但通过 Channel Stacking 处理整个序列。
    输入: (B, T*C, H, W) -> 输出: (B, T*C, H, W)
    用于解决“逐帧生成导致的不一致”问题，模型能一次性看到过去和未来。
    """
    def __init__(self, in_channels=20, out_channels=20, base_filters=64):
        super().__init__()
        
        # --- Encoder ---
        # Level 1 (Original Size)
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_filters, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_filters, base_filters, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # Level 2 (1/2 Size)
        self.down1 = nn.Conv2d(base_filters, base_filters*2, 3, 2, 1) 
        self.enc2 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_filters*2, base_filters*2, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # Level 3 (1/4 Size)
        self.down2 = nn.Conv2d(base_filters*2, base_filters*4, 3, 2, 1)
        self.enc3 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_filters*4, base_filters*4, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # --- Decoder ---
        # Up 1 (1/2 Size)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.reduce1 = nn.Conv2d(base_filters*4 + base_filters*2, base_filters*2, 1, 1, 0) # Skip connect fusion
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_filters*2, base_filters*2, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Up 2 (Original Size)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.reduce2 = nn.Conv2d(base_filters*2 + base_filters, base_filters, 1, 1, 0) # Skip connect fusion
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_filters, base_filters, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Output Layer
        self.final = nn.Conv2d(base_filters, out_channels, 3, 1, 1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)              # (B, 64, H, W)
        e2 = self.enc2(self.down1(e1)) # (B, 128, H/2, W/2)
        e3 = self.enc3(self.down2(e2)) # (B, 256, H/4, W/4)
        
        # Decoder with Skip Connections
        d1 = self.up1(e3)              # (B, 256, H/2, W/2)
        d1 = torch.cat([d1, e2], dim=1)# Concat -> (B, 384, H/2, W/2)
        d1 = self.reduce1(d1)          # -> (B, 128, H/2, W/2)
        d1 = self.dec1(d1)
        
        d2 = self.up2(d1)              # (B, 128, H, W)
        d2 = torch.cat([d2, e1], dim=1)# Concat -> (B, 192, H, W)
        d2 = self.reduce2(d2)          # -> (B, 64, H, W)
        d2 = self.dec2(d2)
        
        out = self.final(d2)
        return out

# ===========================
# 3. Lightning Module (ST-cGAN)
# ===========================
class SimVP_GAN(l.LightningModule):
    def __init__(self, backbone_ckpt_path, lr=2e-4, lambda_adv=1.0, lambda_content=100.0, lambda_fm=10.0):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False 

        # A. 加载骨干网络 (SimVP)
        print(f"[GAN] Loading Backbone from: {backbone_ckpt_path}")
        self.backbone = SimVP.load_from_checkpoint(backbone_ckpt_path)
        self.backbone.freeze() 
        self.backbone.eval()
        self.resize_shape = self.backbone.resize_shape

        # B. 初始化 SequenceRefiner
        # 输入/输出通道 = 20 (序列长度)
        self.refiner = SequenceRefiner(in_channels=20, out_channels=20, base_filters=64)
        
        # C. 初始化 VideoDiscriminator
        # 输入通道 = 1 (单通道降水图)
        self.discriminator = VideoDiscriminator(in_channels=1)

    def forward(self, x):
        # 推理阶段
        with torch.no_grad():
            coarse_pred = self.backbone(x) # (B, T, 1, H, W)
        
        B, T, C, H, W = coarse_pred.shape
        # 转换为序列模式: (B, T*C, H, W) -> (B, 20, 256, 256)
        coarse_seq = coarse_pred.squeeze(2)
        
        # SequenceRefiner 计算残差
        residual = self.refiner(coarse_seq)
        
        fine_seq = coarse_seq + residual
        fine_pred = fine_seq.view(B, T, C, H, W)
        return torch.clamp(fine_pred, 0.0, 1.0)

    def training_step(self, batch, batch_idx):
        opt_g, opt_d = self.optimizers()
        _, x, y, _ = batch
        
        # 确保输入尺寸正确
        x = self.backbone._interpolate_batch_gpu(x, mode='max_pool')
        y = self.backbone._interpolate_batch_gpu(y, mode='max_pool')

        B, T, C, H, W = y.shape
        
        # 准备数据: Video Format (B, C, T, H, W) for Discriminator
        real_video = y.permute(0, 2, 1, 3, 4) # (B, 1, 20, 256, 256)

        # 生成阶段
        with torch.no_grad():
            coarse_pred = self.backbone(x) # (B, T, 1, H, W)
            # 转换为序列模式输入 Refiner: (B, T*1, H, W)
            coarse_seq = coarse_pred.squeeze(2)
        
        # Refiner 修复
        residual = self.refiner(coarse_seq)
        fake_seq = torch.clamp(coarse_seq + residual, 0.0, 1.0) # (B, 20, 256, 256)
        
        # 准备 Fake Video (B, 1, 20, 256, 256)
        fake_video = fake_seq.unsqueeze(1)

        # ==========================
        # 1. 训练判别器 (Video Discriminator)
        # ==========================
        self.toggle_optimizer(opt_d)
        
        # Hinge Loss
        pred_real = self.discriminator(real_video)
        pred_fake = self.discriminator(fake_video.detach())
        
        d_loss = torch.mean(F.relu(1.0 - pred_real)) + torch.mean(F.relu(1.0 + pred_fake))
        
        self.log("train/d_loss", d_loss, prog_bar=True)
        self.manual_backward(d_loss)
        opt_d.step()
        opt_d.zero_grad()
        self.untoggle_optimizer(opt_d)

        # ==========================
        # 2. 训练生成器 (Refiner)
        # ==========================
        self.toggle_optimizer(opt_g)
        
        # 获取判别器特征用于 Feature Matching
        pred_fake, fake_feats = self.discriminator(fake_video, return_feats=True)
        _, real_feats = self.discriminator(real_video, return_feats=True)
        
        # A. 对抗损失 (Adversarial Loss) - 解决模糊
        g_adv_loss = -torch.mean(pred_fake)
        
        # B. 特征匹配损失 (Feature Matching Loss) - 解决模糊/纹理丢失
        g_fm_loss = 0.0
        for feat_f, feat_r in zip(fake_feats, real_feats):
            g_fm_loss += F.l1_loss(feat_f, feat_r)
            
        # C. 掩码内容损失 (Masked Content Loss) - 解决强降水保守
        # 只在有雨的地方(y>0.05)计算L1，让模型别管背景噪声
        # y shape: (B, T, 1, H, W) -> squeeze -> (B, T, H, W)
        target_seq = y.squeeze(2)
        
        rain_mask = (target_seq > 0.05).float() 
        # 对强降水区域加权 (例如 >= 5.0mm, 假设归一化后约 0.16)
        heavy_rain_mask = (target_seq > (5.0/30.0)).float()
        
        pixel_weight = 1.0 + 20.0 * rain_mask + 50.0 * heavy_rain_mask
        
        g_content_loss = torch.mean(torch.abs(fake_seq - target_seq) * pixel_weight)
        
        # 总损失
        g_loss = (self.hparams.lambda_content * g_content_loss) + \
                 (self.hparams.lambda_adv * g_adv_loss) + \
                 (self.hparams.lambda_fm * g_fm_loss)
        
        self.log("train/g_loss", g_loss, prog_bar=True)
        self.log("train/g_content", g_content_loss)
        self.log("train/g_adv", g_adv_loss)
        self.log("train/g_fm", g_fm_loss)
        
        self.manual_backward(g_loss)
        opt_g.step()
        opt_g.zero_grad()
        self.untoggle_optimizer(opt_g)

    def validation_step(self, batch, batch_idx):
        # 验证逻辑
        _, x, y, _ = batch
        x = self.backbone._interpolate_batch_gpu(x, mode='max_pool')
        y = self.backbone._interpolate_batch_gpu(y, mode='max_pool')
        
        y_pred = self(x) # (B, T, 1, H, W)
        
        # 基础指标
        val_mae = F.l1_loss(y_pred, y)
        
        # 计算竞赛评分 (TS Score)
        MM_MAX = 30.0
        pred_mm = torch.clamp(y_pred, 0.0, 1.0) * MM_MAX
        target_mm = y * MM_MAX
        
        thresholds = [0.01, 0.1, 1.0, 2.0, 5.0, 8.0] 
        weights =    [0.1,  0.1, 0.1, 0.2, 0.2, 0.3] 
        
        ts_sum = 0.0
        for t, w in zip(thresholds, weights):
            hits = ((pred_mm >= t) & (target_mm >= t)).float().sum()
            misses = ((pred_mm < t) & (target_mm >= t)).float().sum()
            false_alarms = ((pred_mm >= t) & (target_mm < t)).float().sum()
            
            ts = hits / (hits + misses + false_alarms + 1e-6)
            ts_sum += ts * w
            
        val_score = ts_sum / sum(weights)

        self.log("val_mae", val_mae, on_epoch=True, sync_dist=True)
        self.log("val_score", val_score, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        lr = self.hparams.lr
        # 生成器和判别器使用不同的优化器
        opt_g = torch.optim.Adam(self.refiner.parameters(), lr=lr, betas=(0.5, 0.999))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        return [opt_g, opt_d], []