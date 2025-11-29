# run/train_scwds_mamba.py
import sys, os
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lightning.pytorch.cli import LightningCLI
from metai.dataset import ScwdsDataModule
from metai.model.met_mamba import MeteoMambaModule

def main():
    torch.set_float32_matmul_precision('high')
    
    # Initialize CLI
    cli = LightningCLI(
        MeteoMambaModule, 
        ScwdsDataModule, 
        save_config_callback=None,
        run=True
    )

if __name__ == "__main__":
    main()