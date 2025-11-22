"""
MetAI 配置管理模块

提供统一的配置管理功能，支持：
- YAML配置文件
- 默认值配置
- 配置验证和类型转换
"""

import yaml
from pathlib import Path
from typing import Optional, Union
from dataclasses import dataclass
from metai.utils.met_log import MLOGE, MLOGI

@dataclass
class MetConfig:
    """
    MetAI 配置类
    
    管理开发环境相关的配置参数
    """
    
    # 环境配置
    is_debug: bool = False
    
    # 路径配置
    root_path: str = "/home/dataset-local/SevereWeather_AI_2025"
    
    # 地理数据路径
    # gis_data_path: str = "/home/yyj/code/data/dem"
    gis_data_path: str = "/home/dataset-local/SevereWeather_AI_2025/dem"

    # 文件格式配置
    file_date_format: str = "%m%d-%H%M"

    # NWP文件前缀配置
    nwp_prefix: str = "RRA"

    # 用户ID配置
    user_id: str = 'CP2025000081'
    
    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> 'MetConfig':
        """从YAML配置文件创建配置"""
        config_path = Path(config_path)
        
        if not config_path.exists():
            MLOGE(f"配置文件不存在: {config_path}")
            return cls()
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            # 创建配置对象并更新值
            config = cls()
            for key, value in data.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                else:
                    MLOGE(f"未知配置项: {key}")
            
            return config
            
        except Exception as e:
            MLOGE(f"读取配置文件失败: {e}")
            return cls()
    
    @classmethod
    def load(cls, config_path: Optional[Union[str, Path]] = None, is_debug: Optional[bool] = None) -> 'MetConfig':
        """
        加载配置，根据is_debug自动选择配置文件
        
        Args:
            config_path: 配置文件路径，如果为None则根据is_debug自动选择
            is_debug: 调试模式标志，如果为None则从环境变量或默认值获取
            
        Returns:
            MetConfig: 配置对象
        """
        # 确定is_debug值
        if is_debug is None:
            import os
            is_debug = os.getenv('METAI_DEBUG', 'false').lower() in ('true', '1', 'yes', 'on')
        
        # 查找配置文件
        if config_path is None:
            config_file = "config.yaml"
            
            # 查找配置文件
            default_paths = [
                Path.cwd() / config_file,
                Path.cwd() / "metai" / config_file,
                Path(__file__).parent.parent / config_file,
            ]
            
            for path in default_paths:
                if path.exists():
                    config_path = path
                    break
        
        if config_path:
            config = cls.from_file(config_path)
            config.is_debug = is_debug
            MLOGI(f"从配置文件加载配置: {config_path} (is_debug={is_debug})")
        else:
            config = cls()
            config.is_debug = is_debug
            MLOGI(f"使用默认配置 (is_debug={is_debug})")
        
        return config
    
    def save(self, config_path: Union[str, Path]) -> bool:
        """保存配置到YAML文件"""
        config_path = Path(config_path)
        
        try:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 准备配置数据
            config_data = {}
            for attr_name in dir(self):
                if not attr_name.startswith('_') and not callable(getattr(self, attr_name)):
                    config_data[attr_name] = getattr(self, attr_name)
            
            # 保存YAML文件
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
            
            MLOGI(f"配置已保存到: {config_path}")
            return True
            
        except Exception as e:
            MLOGE(f"保存配置文件失败: {e}")
            return False
    
    def get_root_path(self) -> str:
        """获取数据集根目录路径"""
        return self.root_path
    
    def get_date_format(self) -> str:
        """获取日期格式"""
        return self.file_date_format
    
    def get_nwp_prefix(self) -> str:
        """获取NWP文件前缀"""
        return self.nwp_prefix
    
    def get_user_id(self) -> str:
        """获取用户ID"""
        return self.user_id
    
    def validate(self) -> bool:
        """验证配置的有效性"""
        errors = []
        
        # 验证路径
        if not self.root_path:
            errors.append("root_path 不能为空")
        
        # 验证日期格式
        try:
            from datetime import datetime
            test_date = datetime(2024, 1, 1, 12, 0)
            test_date.strftime(self.file_date_format)
        except ValueError as e:
            errors.append(f"日期格式无效: {e}")
        
        # 验证布尔值
        if not isinstance(self.is_debug, bool):
            errors.append("is_debug 必须是布尔值")
        
        if errors:
            for error in errors:
                MLOGE(f"配置验证错误: {error}")
            return False
        
        return True


# 全局配置实例
_config: Optional[MetConfig] = None


def get_config(config_path: Optional[Union[str, Path]] = None, is_debug: Optional[bool] = None) -> MetConfig:
    """
    获取全局配置实例
    
    Args:
        config_path: 配置文件路径，如果为None则根据is_debug自动选择
        is_debug: 调试模式标志，如果为None则从环境变量或默认值获取
        
    Returns:
        MetConfig: 配置对象
    """
    global _config
    
    # 如果指定了is_debug或config_path，重新加载配置
    if is_debug is not None or config_path is not None or _config is None:
        _config = MetConfig.load(config_path, is_debug)
        if not _config.validate():
            MLOGE("配置验证失败，使用默认配置")
            _config = MetConfig()
            if is_debug is not None:
                _config.is_debug = is_debug
    
    return _config


def reload_config(config_path: Optional[Union[str, Path]] = None, is_debug: Optional[bool] = None) -> MetConfig:
    """
    重新加载配置
    
    Args:
        config_path: 配置文件路径
        is_debug: 调试模式标志
        
    Returns:
        MetConfig: 新的配置对象
    """
    global _config
    _config = MetConfig.load(config_path, is_debug)
    return _config


def create_default_config(config_path: Union[str, Path]) -> bool:
    """
    创建默认配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        bool: 是否创建成功
    """
    config = MetConfig()
    return config.save(config_path)
