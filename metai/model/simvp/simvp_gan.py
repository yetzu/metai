import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as l
from metai.model.simvp import SimVP

# ===========================
# 1. 判别器 (Discriminator) - 保持不变
# ===========================
class Discriminator(nn.Module):
    """
    PatchGAN 风格的判别器，关注局部纹理的真实性
    """
    def __init__(self, in_channels=1):
        super().__init__()
        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.utils.spectral_norm(nn.Conv2d(in_filters, out_filters, 4, 2, 1, bias=False)),
                     nn.LeakyReLU(0.2, inplace=True)]
            return block

        self.model = nn.Sequential(
            # 输入: (B, 1, 256, 256)
            *discriminator_block(in_channels, 64, bn=False),
            # (B, 64, 128, 128)
            *discriminator_block(64, 128),
            # (B, 128, 64, 64)
            *discriminator_block(128, 256),
            # (B, 256, 32, 32)
            *discriminator_block(256, 512),
            # (B, 512, 16, 16)
            nn.Conv2d(512, 1, 4, 1, 1) # 输出 Logits
        )

    def forward(self, img):
        return self.model(img)

# ===========================
# 2. UNet Refiner (核心升级)
# ===========================
class Refiner(nn.Module):
    """
    升级版：基于 UNet 的轻量级残差生成器
    结构：Encoder-Decoder + Skip Connections
    """
    def __init__(self, in_channels=1, out_channels=1, base_filters=32):
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
        e1 = self.enc1(x)              # (B, 32, H, W)
        e2 = self.enc2(self.down1(e1)) # (B, 64, H/2, W/2)
        e3 = self.enc3(self.down2(e2)) # (B, 128, H/4, W/4)
        
        # Decoder with Skip Connections
        d1 = self.up1(e3)              # (B, 128, H/2, W/2)
        d1 = torch.cat([d1, e2], dim=1)# Concat -> (B, 192, H/2, W/2)
        d1 = self.reduce1(d1)          # -> (B, 64, H/2, W/2)
        d1 = self.dec1(d1)
        
        d2 = self.up2(d1)              # (B, 64, H, W)
        d2 = torch.cat([d2, e1], dim=1)# Concat -> (B, 96, H, W)
        d2 = self.reduce2(d2)          # -> (B, 32, H, W)
        d2 = self.dec2(d2)
        
        out = self.final(d2)
        return out

# ===========================
# 3. Lightning Module
# ===========================
class SimVP_GAN(l.LightningModule):
    def __init__(self, backbone_ckpt_path, lr=2e-4, lambda_adv=0.01, lambda_content=1.0):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False 

        # A. 加载骨干网络 (SimVP)
        print(f"[GAN] Loading Backbone from: {backbone_ckpt_path}")
        self.backbone = SimVP.load_from_checkpoint(backbone_ckpt_path)
        self.backbone.freeze() 
        self.backbone.eval()
        self.resize_shape = self.backbone.resize_shape

        # B. 初始化 UNet Refiner
        self.refiner = Refiner(in_channels=1, out_channels=1)
        
        # C. 初始化 Discriminator
        self.discriminator = Discriminator(in_channels=1)

    def forward(self, x):
        with torch.no_grad():
            coarse_pred = self.backbone(x) # (B, T, 1, H, W)
        
        B, T, C, H, W = coarse_pred.shape
        coarse_flat = coarse_pred.view(B * T, C, H, W)
        
        # UNet Refiner 计算残差
        residual = self.refiner(coarse_flat)
        
        fine_flat = coarse_flat + residual
        fine_pred = fine_flat.view(B, T, C, H, W)
        return torch.clamp(fine_pred, 0.0, 1.0)

    def training_step(self, batch, batch_idx):
        opt_g, opt_d = self.optimizers()
        _, x, y, _ = batch
        x = self.backbone._interpolate_batch_gpu(x, mode='max_pool')
        y = self.backbone._interpolate_batch_gpu(y, mode='max_pool')

        B, T, C, H, W = y.shape
        y_flat = y.view(B * T, C, H, W)

        # 生成阶段
        with torch.no_grad():
            coarse_pred = self.backbone(x)
            coarse_flat = coarse_pred.view(B * T, C, H, W)
        
        residual = self.refiner(coarse_flat)
        fake_img = torch.clamp(coarse_flat + residual, 0.0, 1.0)

        # 1. 训练判别器
        self.toggle_optimizer(opt_d)
        real_validity = self.discriminator(y_flat)
        fake_validity = self.discriminator(fake_img.detach())
        d_loss = torch.mean(F.relu(1.0 - real_validity)) + torch.mean(F.relu(1.0 + fake_validity))
        
        self.log("train/d_loss", d_loss, prog_bar=True)
        self.manual_backward(d_loss)
        opt_d.step()
        opt_d.zero_grad()
        self.untoggle_optimizer(opt_d)

        # 2. 训练生成器
        self.toggle_optimizer(opt_g)
        fake_validity_g = self.discriminator(fake_img)
        g_adv_loss = -torch.mean(fake_validity_g)
        g_content_loss = F.l1_loss(fake_img, y_flat)
        
        g_loss = (self.hparams.lambda_content * g_content_loss) + \
                 (self.hparams.lambda_adv * g_adv_loss)
        
        self.log("train/g_loss", g_loss, prog_bar=True)
        self.log("train/g_content", g_content_loss)
        self.log("train/g_adv", g_adv_loss)
        
        self.manual_backward(g_loss)
        opt_g.step()
        opt_g.zero_grad()
        self.untoggle_optimizer(opt_g)

    def validation_step(self, batch, batch_idx):
        _, x, y, _ = batch
        x = self.backbone._interpolate_batch_gpu(x, mode='max_pool')
        y = self.backbone._interpolate_batch_gpu(y, mode='max_pool')
        y_pred = self(x)
        val_mae = F.l1_loss(y_pred, y)
        self.log("val_mae", val_mae, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        lr = self.hparams.lr
        opt_g = torch.optim.Adam(self.refiner.parameters(), lr=lr, betas=(0.5, 0.999))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        return [opt_g, opt_d], []