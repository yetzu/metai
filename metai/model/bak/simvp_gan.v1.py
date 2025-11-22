import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as l
from .simvp_trainer import SimVP

# ===========================
# 1. 判别器 (Discriminator)
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
            # 判别器通常不用 BatchNorm，或者用 InstanceNorm
            # 这里简单起见只用 LeakyReLU 和 SpectralNorm
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
            # (B, 1, 15, 15) -> PatchGAN 输出
        )

    def forward(self, img):
        return self.model(img)

# ===========================
# 2. Refiner (精炼生成器)
# ===========================
class Refiner(nn.Module):
    """
    轻量级残差生成器
    """
    def __init__(self, channels=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, channels, 3, 1, 1)
        )

    def forward(self, x):
        return self.net(x)

# ===========================
# 3. Lightning Module
# ===========================
class SimVP_GAN(l.LightningModule):
    def __init__(self, backbone_ckpt_path, lr=2e-4, lambda_adv=0.01, lambda_content=1.0):
        super().__init__()
        self.save_hyperparameters()
        
        # 关键：GAN 需要手动优化
        self.automatic_optimization = False 

        # A. 加载骨干网络 (SimVP)
        print(f"[GAN] Loading Backbone from: {backbone_ckpt_path}")
        self.backbone = SimVP.load_from_checkpoint(backbone_ckpt_path)
        self.backbone.freeze() # 冻结参数！不计算梯度，极大节省显存
        self.backbone.eval()   # 设置为评估模式 (关闭 Dropout/BN 更新)
        
        # 继承 resize_shape
        self.resize_shape = self.backbone.resize_shape

        # B. 初始化 Refiner
        self.refiner = Refiner(channels=1)
        
        # C. 初始化 Discriminator
        self.discriminator = Discriminator(in_channels=1)

    def forward(self, x):
        # 1. Backbone 推理 (无梯度)
        with torch.no_grad():
            coarse_pred = self.backbone(x) # (B, T, 1, H, W)
        
        # 2. 维度变换: 把 Time 融合进 Batch -> (B*T, 1, H, W)
        B, T, C, H, W = coarse_pred.shape
        coarse_flat = coarse_pred.view(B * T, C, H, W)
        
        # 3. Refiner 计算残差
        residual = self.refiner(coarse_flat)
        
        # 4. 叠加残差
        fine_flat = coarse_flat + residual
        
        # 5. 还原维度并截断
        fine_pred = fine_flat.view(B, T, C, H, W)
        return torch.clamp(fine_pred, 0.0, 1.0)

    def training_step(self, batch, batch_idx):
        opt_g, opt_d = self.optimizers()
        
        _, x, y, _ = batch
        
        # 确保输入尺寸正确 (复用 SimVP 的插值逻辑)
        x = self.backbone._interpolate_batch_gpu(x, mode='max_pool')
        y = self.backbone._interpolate_batch_gpu(y, mode='max_pool')

        B, T, C, H, W = y.shape
        y_flat = y.view(B * T, C, H, W) # 真实图片 (Real)

        # ==========================
        # 生成阶段 (无梯度)
        # ==========================
        with torch.no_grad():
            coarse_pred = self.backbone(x)
            coarse_flat = coarse_pred.view(B * T, C, H, W)
        
        # 计算 Refiner 输出 (需要梯度)
        residual = self.refiner(coarse_flat)
        fake_img = torch.clamp(coarse_flat + residual, 0.0, 1.0) # 生成图片 (Fake)

        # ==========================
        # 1. 训练判别器 (Discriminator)
        # ==========================
        self.toggle_optimizer(opt_d)
        
        # Real Loss
        real_validity = self.discriminator(y_flat)
        # Fake Loss (detach 防止传导给 Generator)
        fake_validity = self.discriminator(fake_img.detach())
        
        # Hinge Loss (推荐用于 GAN)
        d_loss_real = torch.mean(F.relu(1.0 - real_validity))
        d_loss_fake = torch.mean(F.relu(1.0 + fake_validity))
        d_loss = d_loss_real + d_loss_fake
        
        self.log("train/d_loss", d_loss, prog_bar=True)
        self.manual_backward(d_loss)
        opt_d.step()
        opt_d.zero_grad()
        self.untoggle_optimizer(opt_d)

        # ==========================
        # 2. 训练生成器 (Refiner)
        # ==========================
        self.toggle_optimizer(opt_g)
        
        # 对抗损失: 骗过判别器 (希望 fake_validity 越大越好)
        fake_validity_g = self.discriminator(fake_img)
        g_adv_loss = -torch.mean(fake_validity_g)
        
        # 内容损失: 保证位置不偏离 (L1)
        g_content_loss = F.l1_loss(fake_img, y_flat)
        
        # 总损失
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
        # 验证时只看生成效果
        _, x, y, _ = batch
        x = self.backbone._interpolate_batch_gpu(x, mode='max_pool')
        y = self.backbone._interpolate_batch_gpu(y, mode='max_pool')
        
        # Forward (包含 Backbone + Refiner)
        y_pred = self(x)
        
        val_mae = F.l1_loss(y_pred, y)
        val_mse = F.mse_loss(y_pred, y)
        self.log("val_mae", val_mae, on_epoch=True, sync_dist=True)
        self.log("val_mse", val_mse, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        lr = self.hparams.lr
        # 生成器和判别器使用不同的优化器
        opt_g = torch.optim.Adam(self.refiner.parameters(), lr=lr, betas=(0.5, 0.999))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        return [opt_g, opt_d], []