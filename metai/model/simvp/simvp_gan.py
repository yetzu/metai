# metai/model/simvp/simvp_gan.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as l
from metai.model.simvp import SimVP

# ===========================
# 1. 3D 视频判别器 (Video Discriminator) - 保持不变
# ===========================
class VideoDiscriminator(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        
        def disc_block(in_f, out_f, kernel_size=(3,4,4), stride=(1,2,2), padding=(1,1,1), bn=True):
            layers = [nn.utils.spectral_norm(nn.Conv3d(in_f, out_f, kernel_size, stride, padding, bias=False))]
            if bn:
                layers.append(nn.InstanceNorm3d(out_f))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.ModuleList([
            nn.Sequential(*disc_block(in_channels, 32, bn=False)),
            nn.Sequential(*disc_block(32, 64)),
            nn.Sequential(*disc_block(64, 128)),
            nn.Sequential(*disc_block(128, 256)),
        ])
        
        self.final_conv = nn.Conv3d(256, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x, return_feats=False):
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
# 2. 时序感知 Refiner - 保持不变
# ===========================
class SequenceRefiner(nn.Module):
    def __init__(self, in_channels=20, out_channels=20, base_filters=64):
        super().__init__()
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_filters, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_filters, base_filters, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.down1 = nn.Conv2d(base_filters, base_filters*2, 3, 2, 1) 
        
        self.enc2 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_filters*2, base_filters*2, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.down2 = nn.Conv2d(base_filters*2, base_filters*4, 3, 2, 1)
        
        self.enc3 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_filters*4, base_filters*4, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Decoder
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.reduce1 = nn.Conv2d(base_filters*4 + base_filters*2, base_filters*2, 1, 1, 0)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_filters*2, base_filters*2, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.reduce2 = nn.Conv2d(base_filters*2 + base_filters, base_filters, 1, 1, 0)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_filters, base_filters, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.final = nn.Conv2d(base_filters, out_channels, 3, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)              
        e2 = self.enc2(self.down1(e1)) 
        e3 = self.enc3(self.down2(e2)) 
        
        d1 = self.up1(e3)              
        d1 = torch.cat([d1, e2], dim=1)
        d1 = self.reduce1(d1)          
        d1 = self.dec1(d1)
        
        d2 = self.up2(d1)              
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.reduce2(d2)          
        d2 = self.dec2(d2)
        
        out = self.final(d2)
        return out

# ===========================
# 3. Lightning Module (ST-cGAN) - 支持课程学习
# ===========================
class SimVP_GAN(l.LightningModule):
    def __init__(self, backbone_ckpt_path, lr=1e-4, 
                 # 初始参数 (Safe Mode)
                 lambda_adv=0.01, 
                 lambda_content=1000.0, 
                 lambda_fm=10.0,
                 # 课程学习配置
                 use_curriculum=True,
                 curr_start_epoch=10,       # 第 10 个 Epoch 开始变化
                 curr_transition_epochs=10, # 过渡期 10 个 Epoch (10->20)
                 target_lambda_adv=0.1,     # 最终目标：增强对抗 (0.01 -> 0.1)
                 target_lambda_content=100.0, # 最终目标：放宽约束 (1000 -> 100)
                 target_lambda_fm=20.0      # 最终目标：增强纹理 (10 -> 20)
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False 

        # A. 加载 SimVP Backbone
        print(f"[GAN] Loading Backbone from: {backbone_ckpt_path}")
        self.backbone = SimVP.load_from_checkpoint(backbone_ckpt_path)
        self.backbone.freeze() 
        self.backbone.eval()
        self.resize_shape = self.backbone.resize_shape

        # B. 初始化组件
        self.refiner = SequenceRefiner(in_channels=20, out_channels=20, base_filters=64)
        self.discriminator = VideoDiscriminator(in_channels=1)

        # C. 初始化当前权重 (Instance Variables)
        self.curr_adv = lambda_adv
        self.curr_content = lambda_content
        self.curr_fm = lambda_fm

    def on_train_epoch_start(self):
        """课程学习核心逻辑：动态调整权重"""
        if not self.hparams.use_curriculum:
            return

        epoch = self.current_epoch
        start = self.hparams.curr_start_epoch
        duration = self.hparams.curr_transition_epochs
        
        # 计算进度 (0.0 -> 1.0)
        if epoch < start:
            progress = 0.0
        elif epoch >= start + duration:
            progress = 1.0
        else:
            progress = (epoch - start) / duration
            
        # 线性插值更新权重
        self.curr_adv = self.hparams.lambda_adv + progress * (self.hparams.target_lambda_adv - self.hparams.lambda_adv)
        self.curr_content = self.hparams.lambda_content + progress * (self.hparams.target_lambda_content - self.hparams.lambda_content)
        self.curr_fm = self.hparams.lambda_fm + progress * (self.hparams.target_lambda_fm - self.hparams.lambda_fm)
        
        # 打印日志 (防止刷屏，仅在关键节点或每轮开始打印)
        if self.trainer.is_global_zero:
            print(f"\n[Curriculum] Epoch {epoch} (Progress {progress:.2f}): "
                  f"Content={self.curr_content:.1f}, Adv={self.curr_adv:.3f}, FM={self.curr_fm:.1f}")
            
        # 记录到 TensorBoard (使用 sync_dist=True 确保分布式训练中的同步)
        self.log("train/weight_content", self.curr_content, on_epoch=True, sync_dist=True)
        self.log("train/weight_adv", self.curr_adv, on_epoch=True, sync_dist=True)
        self.log("train/weight_fm", self.curr_fm, on_epoch=True, sync_dist=True)

    def forward(self, x):
        # 推理逻辑
        with torch.no_grad():
            coarse_logits = self.backbone(x) 
            coarse_pred = torch.sigmoid(coarse_logits) 
        
        B, T, C, H, W = coarse_pred.shape
        coarse_seq = coarse_pred.squeeze(2) 
        
        residual = self.refiner(coarse_seq)
        
        fine_seq = coarse_seq + residual
        fine_pred = fine_seq.view(B, T, C, H, W)
        return torch.clamp(fine_pred, 0.0, 1.0)

    def training_step(self, batch, batch_idx):
        opt_g, opt_d = self.optimizers()
        # 解包
        _, x, y, _, _ = batch
        
        x = self.backbone._interpolate_batch_gpu(x, mode='max_pool')
        y = self.backbone._interpolate_batch_gpu(y, mode='max_pool')

        # 准备真值视频
        real_video = y.permute(0, 2, 1, 3, 4) 

        # === 生成阶段 ===
        with torch.no_grad():
            coarse_logits = self.backbone(x) 
            coarse_seq = torch.sigmoid(coarse_logits).squeeze(2)
        
        residual = self.refiner(coarse_seq)
        fake_seq = torch.clamp(coarse_seq + residual, 0.0, 1.0)
        fake_video = fake_seq.unsqueeze(1).permute(0, 1, 2, 3, 4)

        # ==========================
        # 1. 训练判别器
        # ==========================
        self.toggle_optimizer(opt_d)
        pred_real = self.discriminator(real_video)
        pred_fake = self.discriminator(fake_video.detach())
        d_loss = torch.mean(F.relu(1.0 - pred_real)) + torch.mean(F.relu(1.0 + pred_fake))
        
        self.log("train/d_loss", d_loss, prog_bar=True)
        self.manual_backward(d_loss)
        opt_d.step()
        opt_d.zero_grad()
        self.untoggle_optimizer(opt_d)

        # ==========================
        # 2. 训练生成器 (使用动态权重)
        # ==========================
        self.toggle_optimizer(opt_g)
        pred_fake, fake_feats = self.discriminator(fake_video, return_feats=True)
        _, real_feats = self.discriminator(real_video, return_feats=True)
        
        # Loss A: Adversarial
        g_adv_loss = -torch.mean(pred_fake)
        
        # Loss B: Feature Matching
        g_fm_loss = 0.0
        for feat_f, feat_r in zip(fake_feats, real_feats):
            g_fm_loss += F.l1_loss(feat_f, feat_r)
            
        # Loss C: Masked Content Loss
        target_seq = y.squeeze(2)
        rain_mask = (target_seq > 0.05).float() 
        heavy_rain_mask = (target_seq > (5.0/30.0)).float()
        pixel_weight = 1.0 + 20.0 * rain_mask + 50.0 * heavy_rain_mask
        g_content_loss = torch.mean(torch.abs(fake_seq - target_seq) * pixel_weight)
        
        # 🚨 关键点：使用 self.curr_* 动态权重
        g_loss = (self.curr_content * g_content_loss) + \
                 (self.curr_adv * g_adv_loss) + \
                 (self.curr_fm * g_fm_loss)
        
        self.log("train/g_loss", g_loss, prog_bar=True)
        self.log("train/g_content", g_content_loss)
        self.log("train/g_adv", g_adv_loss)
        
        self.manual_backward(g_loss)
        opt_g.step()
        opt_g.zero_grad()
        self.untoggle_optimizer(opt_g)

    def validation_step(self, batch, batch_idx):
        _, x, y, _, _ = batch
        x = self.backbone._interpolate_batch_gpu(x, mode='max_pool')
        y = self.backbone._interpolate_batch_gpu(y, mode='max_pool')
        y_pred = self(x) 
        val_mae = F.l1_loss(y_pred, y)
        
        # TS Score 计算
        MM_MAX = 30.0
        pred_mm = y_pred * MM_MAX
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
        opt_g = torch.optim.Adam(self.refiner.parameters(), lr=lr, betas=(0.5, 0.999))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        return [opt_g, opt_d], []