from pydantic import BaseModel, Field, ConfigDict
from typing import Tuple

class MeteoMambaConfig(BaseModel):
    # 路径与基础
    data_path: str = "data/samples.jsonl"
    save_dir: str = "./output/mamba"
    
    # 形状参数
    in_shape: Tuple[int, int, int, int] = (10, 31, 256, 256)
    aft_seq_length: int = 20
    
    # 模型参数
    hid_S: int = 64
    hid_T: int = 256
    N_S: int = 4
    N_T: int = 8
    
    # 训练参数
    batch_size: int = 4
    accumulate_grad_batches: int = 4 
    lr: float = 5e-4
    min_lr: float = 1e-5
    max_epochs: int = 100
    warmup_epoch: int = 10
    
    # 策略
    use_curriculum_learning: bool = True
    
    model_config = ConfigDict(protected_namespaces=())
    
    def to_dict(self):
        return self.model_dump()