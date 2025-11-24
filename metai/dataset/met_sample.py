import os
import math
from dataclasses import dataclass, field
from functools import cached_property
from typing import List, Optional, Dict, Union, Tuple
from datetime import datetime, timedelta
import numpy as np
import cv2
from metai.utils import get_config
from metai.utils import MetConfig, MetLabel, MetRadar, MetNwp, MetGis, MetVar

# 默认通道列表常量 - 保持不变
_DEFAULT_CHANNELS: List[Union[MetLabel, MetRadar, MetNwp, MetGis]] = [
    MetLabel.RA, MetRadar.CR, MetRadar.CAP20, MetRadar.CAP30, MetRadar.CAP40, MetRadar.CAP50, MetRadar.CAP60, MetRadar.CAP70, MetRadar.ET, MetRadar.HBR, MetRadar.VIL,
    MetNwp.WS925, MetNwp.WS700, MetNwp.WS500, MetNwp.Q1000, MetNwp.Q850, MetNwp.Q700, MetNwp.PWAT, MetNwp.PE, MetNwp.TdSfc850, MetNwp.LCL, MetNwp.KI, MetNwp.CAPE,
    MetGis.LAT, MetGis.LON, MetGis.DEM, MetGis.MONTH_SIN, MetGis.MONTH_COS, MetGis.HOUR_SIN, MetGis.HOUR_COS
]


@dataclass
class MetSample:
    """
    天气样本数据结构，基于竞赛数据目录结构设计。
    用于管理单个天气样本的元数据和配置信息。
    """
    sample_id: str
    timestamps: List[str]
    met_config: MetConfig
    is_debug: bool = field(default_factory=lambda: False)
    is_train: bool = field(default_factory=lambda: True)
    test_set: str = field(default_factory=lambda: "TestSet")
    
    channels: List[Union[MetLabel, MetRadar, MetNwp, MetGis]] = field(
        default_factory=lambda: _DEFAULT_CHANNELS.copy()
    )
    channel_size: int = field(default_factory=lambda: len(_DEFAULT_CHANNELS))
    
    task_mode: str = field(default_factory=lambda: 'precipitation')
    default_shape: Tuple[int, int] = field(default_factory=lambda: (301, 301))
    max_timesteps: int = field(default_factory=lambda: 10)
    # [新增] 明确定义输出帧数为 20
    out_timesteps: int = field(default_factory=lambda: 20)

    _gis_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = field(default_factory=dict, init=False, repr=False)
    _sample_id_parts: Optional[List[str]] = field(default=None, init=False, repr=False)

    @classmethod
    def create(cls, sample_id: str, timestamps: List[str], config: Optional['MetConfig'] = None, **kwargs) -> 'MetSample':
        if config is None:
            config = get_config(is_debug=kwargs.get('is_debug', False))
        
        if 'is_debug' in kwargs:
            config.is_debug = kwargs['is_debug']
        else:
            kwargs['is_debug'] = config.is_debug

        return cls(
            sample_id=sample_id,
            timestamps=timestamps,
            met_config=config,
            **kwargs
        )
    
    def _get_sample_id_parts(self) -> List[str]:
        if self._sample_id_parts is None:
            self._sample_id_parts = self.sample_id.split('_')
        return self._sample_id_parts
    
    @cached_property
    def task_id(self) -> str:
        return self._get_sample_id_parts()[0]

    @cached_property
    def region_id(self) -> str:
        return self._get_sample_id_parts()[1]

    @cached_property
    def time_id(self) -> str:
        return self._get_sample_id_parts()[2]

    @cached_property
    def station_id(self) -> str:
        return self._get_sample_id_parts()[3]

    @cached_property
    def radar_type(self) -> str:
        return self._get_sample_id_parts()[4]

    @cached_property
    def batch_id(self) -> str:
        return self._get_sample_id_parts()[5]

    @cached_property
    def case_id(self) -> str:
        parts = self._get_sample_id_parts()
        return '_'.join(parts[:4])

    @cached_property
    def base_path(self) -> str:
        return os.path.join(
            self.met_config.root_path,
            self.task_id,
            "TrainSet" if self.is_train else self.test_set,
            self.region_id,
            self.case_id,
        )
    
    @property
    def root_path(self) -> str:
        return self.met_config.root_path
    
    @property
    def gis_data_path(self) -> str:
        return self.met_config.gis_data_path
    
    @property
    def file_date_format(self) -> str:
        return self.met_config.file_date_format
    
    @property
    def nwp_prefix(self) -> str:
        return self.met_config.nwp_prefix
    
    @property
    def user_id(self) -> str:
        return self.met_config.user_id
    
    @property
    def metadata(self) -> Dict:
        metadata_dict = vars(self).copy()
        metadata_dict.pop('_gis_cache', None)
        return metadata_dict

    def str_to_datetime(self, time_str: str) -> datetime:
        return datetime.strptime(time_str, self.file_date_format)

    def datetime_to_str(self, datetime_obj: datetime) -> str:
        return datetime_obj.strftime(self.file_date_format)

    def normalize(self, file_path: str, min_value: float = 0, max_value: float = 300) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        数据预处理：裁剪异常值并归一化。返回归一化后的数据和有效值 Mask。
        
        Returns:
            tuple: (归一化后的数据, 有效值mask)
                - 归一化后的数据: 范围在[0, 1]之间，float32类型
                - 有效值mask: True表示有效值（非NaN且文件存在），bool类型
        """
        try:
            data = np.load(file_path)
            
            # 提取有效值 mask：文件成功加载且数据是有限值
            valid_mask = np.isfinite(data)
            
            # 归一化处理（缺测值会被处理）
            data = np.nan_to_num(data, nan=min_value, neginf=min_value, posinf=max_value)
            inv_denom = 1.0 / (max_value - min_value)
            scale = (data - min_value) * inv_denom
            np.clip(scale, 0.0, 1.0, out=scale)
            
            return scale.astype(np.float32, copy=False), valid_mask.astype(bool, copy=False)
        except Exception:
            return None, None

    def normalize_with_mask(self, file_path: str, min_value: float = 0, max_value: float = 300, missing_value: float = -9) -> tuple[np.ndarray, np.ndarray]:
        """
        数据预处理：裁剪异常值并归一化，同时提取目标数据的严格有效值mask（用于目标数据）。
        
        Returns:
            tuple: (归一化后的数据, 有效值mask)
        """
        try:
            data = np.load(file_path)
            # 提取有效值mask（在归一化之前）：排除 missing_value 和非有限值
            valid_mask = (data != missing_value) & np.isfinite(data)
            
            # 归一化处理（缺测值会被处理）
            data = np.nan_to_num(data, nan=min_value, neginf=min_value, posinf=max_value)
            inv_denom = 1.0 / (max_value - min_value)
            scale = (data - min_value) * inv_denom
            np.clip(scale, 0.0, 1.0, out=scale)
            
            return scale.astype(np.float32, copy=False), valid_mask.astype(bool, copy=False)
        except Exception:
            default_data = np.zeros(self.default_shape, dtype=np.float32)
            default_mask = np.zeros(self.default_shape, dtype=bool)
            return default_data, default_mask
    
    def load_input_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        [SOTA 改进] 加载输入数据，采用 Early Fusion 策略融合未来 NWP。
        结构: 
        1. 基础输入 (10帧): [Radar, Past_NWP, GIS]
        2. 未来引导 (20帧折叠为10帧): [Future_NWP]
        
        Returns:
            input_data: (T=10, C_total, H, W)
            input_mask: (T=10, C_total, H, W)
        """
        # 1. 定义时间范围
        # 输入时间: T=0 ~ T=9 (过去1小时)
        past_timesteps = self.timestamps[:self.max_timesteps] 
        # 未来引导时间: T=10 ~ T=29 (未来2小时) -> 对应输出的时刻
        future_start = self.max_timesteps
        future_end = future_start + self.out_timesteps
        future_timesteps = self.timestamps[future_start:future_end]
        
        # 预加载 GIS
        self._preload_gis_data()

        # --- A. 加载基础输入 (Past 10 Frames) ---
        past_series = []
        past_masks = []
        
        for i, timestamp in enumerate(past_timesteps):
            frames = []
            masks = []
            for channel in self.channels:
                data, mask = self._load_channel_frame_with_fallback(channel, timestamp, i, self.timestamps)
                frames.append(data)
                masks.append(mask)
            past_series.append(np.stack(frames, axis=0))
            past_masks.append(np.stack(masks, axis=0))
            
        # Shape: (10, C_base, H, W)
        input_base = np.stack(past_series, axis=0)
        mask_base = np.stack(past_masks, axis=0)

        # --- B. 加载未来 NWP 引导 (Future 20 Frames) ---
        # 筛选出 NWP 通道
        nwp_channels = [c for c in self.channels if isinstance(c, MetNwp)]
        
        if len(nwp_channels) > 0 and len(future_timesteps) == self.out_timesteps:
            future_nwp_series = []
            future_nwp_masks = []
            
            # 加载 20 帧未来 NWP
            for i, timestamp in enumerate(future_timesteps):
                frames = []
                masks = []
                for channel in nwp_channels:
                    # 注意：这里我们只加载 NWP
                    # 对于未来数据，如果缺测，也尝试 fallback，或者补0
                    # 由于 _load_channel_frame_with_fallback 需要全局索引，这里简单处理直接调 _load_nwp_frame
                    # 或者复用 fallback 逻辑，注意 index 偏移
                    global_idx = future_start + i
                    data, mask = self._load_channel_frame_with_fallback(channel, timestamp, global_idx, self.timestamps)
                    frames.append(data)
                    masks.append(mask)
                future_nwp_series.append(np.stack(frames, axis=0))
                future_nwp_masks.append(np.stack(masks, axis=0))
            
            # Shape: (20, C_nwp, H, W)
            input_future_nwp = np.stack(future_nwp_series, axis=0)
            mask_future_nwp = np.stack(future_nwp_masks, axis=0)
            
            # --- C. 时间折叠 (Temporal Folding) ---
            # 将 20 帧折叠为 10 帧: (20, C, H, W) -> (10, 2, C, H, W) -> (10, 2*C, H, W)
            # 这样第 i 帧输入就包含了第 2i 和 2i+1 帧的未来指引
            B_t, C_n, H_t, W_t = input_future_nwp.shape
            fold_factor = self.out_timesteps // self.max_timesteps # 20 // 10 = 2
            
            if self.out_timesteps % self.max_timesteps == 0:
                input_folded = input_future_nwp.reshape(self.max_timesteps, fold_factor * C_n, H_t, W_t)
                mask_folded = mask_future_nwp.reshape(self.max_timesteps, fold_factor * C_n, H_t, W_t)
                
                # --- D. 拼接 ---
                # Final: (10, C_base + 2*C_nwp, H, W)
                input_data = np.concatenate([input_base, input_folded], axis=1)
                input_mask = np.concatenate([mask_base, mask_folded], axis=1)
            else:
                # 如果不能整除 (极少见)，直接放弃拼接未来 NWP，避免报错
                input_data = input_base
                input_mask = mask_base
        else:
            # 如果没有 NWP 通道或未来时间步不足
            input_data = input_base
            input_mask = mask_base

        return input_data, input_mask

    def load_target_data(self) -> tuple[np.ndarray, np.ndarray]:
        target_data = []
        valid_mask = []

        start_idx = self.max_timesteps
        end_idx = start_idx + self.out_timesteps
        target_timestamps = self.timestamps[start_idx : end_idx]

        for timestamp in target_timestamps:
            file_path = os.path.join(
                self.base_path,
                MetVar.LABEL.name,
                MetLabel.RA.name,
                f"CP_Label_{MetLabel.RA.name}_{self.station_id}_{timestamp}.npy"
            )

            min_val, max_val = self._get_channel_limits(MetLabel.RA)
            missing_val = getattr(MetLabel.RA, 'missing_value', -9.0)
            data, mask = self.normalize_with_mask(file_path, min_val, max_val, missing_value=missing_val)
            target_data.append(data)
            valid_mask.append(mask)

        target_data = np.expand_dims(np.stack(target_data, axis=0), axis=1)  # (T, C, H, W)
        valid_mask = np.expand_dims(np.stack(valid_mask, axis=0), axis=1)  # (T, C, H, W)
        return target_data, valid_mask

    def to_numpy(self, is_train: bool=False) -> Tuple[Dict, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        将数据转换为numpy数组
        
        Returns:
            Tuple: (metadata, input_data, target_data, target_mask, input_mask)
                - metadata: 样本元数据字典
                - input_data: 输入数据，形状为 (T, C, H, W)
                - target_data: 目标数据，形状为 (T_out, C, H, W)，测试时为 None
                - target_mask: 目标有效值mask，形状为 (T_out, C, H, W)，测试时为 None
                - input_mask: 输入有效值mask，形状为 (T_in, C, H, W)
        """
        input_data, input_mask = self.load_input_data()
        
        if is_train:
            target_data, target_mask = self.load_target_data()
            return self.metadata, input_data, target_data, target_mask, input_mask
        else:
            return self.metadata, input_data, None, None, input_mask
    
    def _get_channel_limits(self, channel: Union[MetLabel, MetRadar, MetNwp, MetGis]) -> tuple[float, float]:
        return (
            float(getattr(channel, "min", 0.0)),
            float(getattr(channel, "max", 1.0))
        )

    def _load_label_frame(self, var: MetLabel, timestamp: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        file_path = os.path.join(
            self.base_path, var.parent, var.value, 
            f"{self.task_id}_Label_{var.value}_{self.station_id}_{timestamp}.npy"
        )
        min_val, max_val = self._get_channel_limits(var)
        return self.normalize(file_path, min_val, max_val)

    def _load_radar_frame(self, var: MetRadar, timestamp: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        file_path = os.path.join(
            self.base_path, var.parent, var.value, 
            f"{self.task_id}_RADA_{self.station_id}_{timestamp}_{self.radar_type}_{var.value}.npy"
        )
        min_val, max_val = self._get_channel_limits(var)
        return self.normalize(file_path, min_val, max_val)

    def _load_nwp_frame(self, var: MetNwp, timestamp: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]: # type: ignore
        obs_time = self.str_to_datetime(timestamp)

        if obs_time.minute < 30:
            obs_time = obs_time.replace(minute=0)
        else:
            obs_time = obs_time.replace(minute=0) + timedelta(hours=1)

        file_path = os.path.join(
            self.base_path, var.parent, var.name, 
            f"{self.task_id}_{self.nwp_prefix}_{self.station_id}_{self.datetime_to_str(obs_time)}_{var.value}.npy"
        )
        min_val, max_val = self._get_channel_limits(var)
        data, mask = self.normalize(file_path, min_val, max_val)
        
        if data is None or mask is None:
            return None, None
            
        # NWP 数据需要进行插值
        dsize = (self.default_shape[1], self.default_shape[0])
        resized_data = cv2.resize(data, dsize, interpolation=cv2.INTER_LINEAR)
        resized_mask = cv2.resize(mask.astype(np.uint8), dsize, interpolation=cv2.INTER_NEAREST).astype(bool)
        
        return resized_data.astype(np.float32, copy=False), resized_mask

    def _load_gis_frame(self, var: MetGis, timestamp: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        加载GIS数据帧。
        对于静态数据(DEM/LAT/LON)，从文件加载；
        对于时间数据(MONTH/HOUR)，采用周期性编码 (Sin/Cos) 动态生成。
        """
        # GIS数据通常被认为是全局有效的，初始 Mask 全为 True
        full_mask = np.full(self.default_shape, True, dtype=bool)
        
        # --- 处理周期性时间变量 ---
        if var in [MetGis.MONTH_SIN, MetGis.MONTH_COS, MetGis.HOUR_SIN, MetGis.HOUR_COS]:
            try:
                obs_time = self.str_to_datetime(timestamp)
                raw_val = 0.0
                
                # 计算周期性数值
                if var in [MetGis.MONTH_SIN, MetGis.MONTH_COS]:
                    # 月份周期为 12
                    # 将 1-12 映射到角度。 (month / 12) * 2pi
                    angle = 2 * math.pi * float(obs_time.month) / 12.0
                    raw_val = math.sin(angle) if var == MetGis.MONTH_SIN else math.cos(angle)
                    
                elif var in [MetGis.HOUR_SIN, MetGis.HOUR_COS]:
                    # 小时周期为 24
                    # 将 0-23 映射到角度。 (hour / 24) * 2pi
                    angle = 2 * math.pi * float(obs_time.hour) / 24.0
                    raw_val = math.sin(angle) if var == MetGis.HOUR_SIN else math.cos(angle)
                
                # 归一化处理
                # MetGis 定义中 min=-1, max=1
                # 归一化公式: (x - min) / (max - min) => (x - (-1)) / 2 => (x + 1) / 2
                # 结果将映射到 [0, 1] 区间，符合网络输入规范
                min_val, max_val = self._get_channel_limits(var)
                if max_val - min_val > 0:
                    normalized_value = (raw_val - min_val) / (max_val - min_val)
                else:
                    normalized_value = 0.5 # 默认中间值
                
                # 裁剪保险
                normalized_value = max(0.0, min(1.0, normalized_value))
                
                # 填充整个空间网格
                data = np.full(self.default_shape, normalized_value, dtype=np.float32)
                return data, full_mask
                
            except (ValueError, AttributeError, Exception) as e:
                # 异常兜底
                return np.zeros(self.default_shape, dtype=np.float32), np.zeros(self.default_shape, dtype=bool)

        # --- 处理静态 GIS 文件数据 (LAT, LON, DEM) ---
        try:
            # 文件名映射：MetGis.LAT.value -> "lat.npy"
            file_name = f"{var.value}.npy"
            file_path = os.path.join(self.gis_data_path, self.station_id, file_name)
            
            min_val, max_val = self._get_channel_limits(var)
            
            # 使用通用的 normalize 方法加载并归一化
            data, mask = self.normalize(file_path, min_val, max_val)
            
            if data is None:
                return np.zeros(self.default_shape, dtype=np.float32), np.zeros(self.default_shape, dtype=bool)
                
            return data, mask

        except Exception:
            return np.zeros(self.default_shape, dtype=np.float32), np.zeros(self.default_shape, dtype=bool)

    def _preload_gis_data(self):
        """预加载所有GIS数据到缓存"""
        self._gis_cache.clear()
        # 创建默认零填充和全 False mask
        default_data = np.zeros(self.default_shape, dtype=np.float32)
        default_mask = np.zeros(self.default_shape, dtype=bool)
        
        for channel in self.channels:
            if isinstance(channel, MetGis):
                cache_key = f"gis_{channel.value}"
                if cache_key not in self._gis_cache:
                    data, mask = self._load_gis_frame(channel, self.timestamps[0])
                    self._gis_cache[cache_key] = (data if data is not None else default_data, 
                                                 mask if mask is not None else default_mask)

    def _load_channel_frame(self, channel: Union[MetLabel, MetRadar, MetNwp, MetGis], timestamp: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        加载单个通道在指定时刻的数据和 Mask。
        返回 (None, None) 表示数据缺失，便于后续缺测补值处理。
        """
        loader_map = {
            MetLabel: self._load_label_frame,
            MetRadar: self._load_radar_frame,
            MetGis: self._load_gis_frame,
            MetNwp: self._load_nwp_frame,
        }

        loader = None
        for enum_cls, handler in loader_map.items():
            if isinstance(channel, enum_cls):
                loader = handler
                break

        if loader is None:
            return None, None

        data, mask = loader(channel, timestamp)

        if data is None:
            return None, None

        # 确保数据类型正确
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float32)
        if not isinstance(mask, np.ndarray):
            mask = np.array(mask, dtype=bool)

        return data.astype(np.float32, copy=False), mask.astype(bool, copy=False)

    def _load_temporal_data_with_fallback(self, channel: Union[MetLabel, MetRadar, MetNwp, MetGis], timestamp: str, 
                                         timestep_idx: int, all_timestamps: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        加载时序数据，支持缺测补值策略：当前时次缺测时，按前一时次 → 后一时次 → 补0。
        返回数据和 Mask。如果发生补值，Mask 对应的位置为 False。
        """
        # 创建默认零填充和全 False mask
        default_data = np.zeros(self.default_shape, dtype=np.float32)
        default_mask = np.zeros(self.default_shape, dtype=bool)

        # 1. 尝试加载当前时次数据
        data, mask = self._load_channel_frame(channel, timestamp)
        if data is not None and mask is not None:
            return data, mask
        
        # 2. 当前时次缺测，使用前一时次补值
        if timestep_idx > 0:
            prev_timestamp = all_timestamps[timestep_idx - 1]
            data, mask = self._load_channel_frame(channel, prev_timestamp)
            if data is not None:
                return data, default_mask
        
        # 3. 前一时次也缺测，使用后一时次补值
        if timestep_idx < len(all_timestamps) - 1:
            next_timestamp = all_timestamps[timestep_idx + 1]
            data, mask = self._load_channel_frame(channel, next_timestamp)
            if data is not None:
                return data, default_mask
        
        return default_data, default_mask

    def _load_channel_frame_with_fallback(self, channel: Union[MetLabel, MetRadar, MetNwp, MetGis], timestamp: str, 
                                         timestep_idx: int, all_timestamps: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        加载单个通道数据，支持缺测补值策略。
        """
        # 创建默认零填充和全 False mask
        default_data = np.zeros(self.default_shape, dtype=np.float32)
        default_mask = np.zeros(self.default_shape, dtype=bool)

        # 1. GIS数据直接从缓存读取
        if isinstance(channel, MetGis):
            cache_key = f"gis_{channel.value}"
            if cache_key in self._gis_cache:
                return self._gis_cache[cache_key]
            # 理论上应该已经被预加载，如果缓存中没有，尝试加载并返回默认零填充/全 False mask
            data, mask = self._load_gis_frame(channel, timestamp)
            if data is not None and mask is not None:
                self._gis_cache[cache_key] = (data, mask)
                return data, mask
            return default_data, default_mask
        
        if isinstance(channel, (MetRadar, MetNwp)):
            return self._load_temporal_data_with_fallback(channel, timestamp, timestep_idx, all_timestamps)
        
        data, mask = self._load_channel_frame(channel, timestamp)
        if data is not None and mask is not None:
            return data, mask
        
        return default_data, default_mask