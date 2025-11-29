import os
import math
import lmdb
import zlib
import pickle
import numpy as np
import cv2
from dataclasses import dataclass, field
from functools import cached_property
from typing import List, Optional, Dict, Union, Tuple
from datetime import datetime, timedelta

# 假设这些工具类在您的环境中可用
from metai.utils import get_config
from metai.utils import MetConfig, MetLabel, MetRadar, MetNwp, MetGis, MetVar

# ==============================================================================
# 通道定义
# ==============================================================================
_DEFAULT_CHANNELS: List[Union[MetLabel, MetRadar, MetNwp, MetGis]] = [
    MetLabel.RA, 
    MetRadar.CR, MetRadar.CAP30, MetRadar.CAP50, MetRadar.ET, MetRadar.VIL,
    MetNwp.WS925, MetNwp.WS500, MetNwp.Q850, MetNwp.Q700, MetNwp.PWAT, MetNwp.CAPE,
    MetGis.LAT, MetGis.LON, MetGis.DEM, 
    MetGis.MONTH_SIN, MetGis.MONTH_COS, MetGis.HOUR_SIN, MetGis.HOUR_COS
]

@dataclass
class MetSample:
    """
    天气样本数据结构
    """
    sample_id: str
    timestamps: List[str]
    met_config: MetConfig
    
    is_train: bool = field(default_factory=lambda: True)
    test_set: str = field(default_factory=lambda: "TestSet")
    
    lmdb_root: str = field(default_factory=lambda: "/data/zjobs/LMDB")

    channels: List[Union[MetLabel, MetRadar, MetNwp, MetGis]] = field(
        default_factory=lambda: _DEFAULT_CHANNELS.copy()
    )
    channel_size: int = field(default_factory=lambda: len(_DEFAULT_CHANNELS))
    
    default_shape: Tuple[int, int] = field(default_factory=lambda: (256, 256))
    
    # 序列长度定义
    obs_seq_len: int = field(default_factory=lambda: 10)
    pred_seq_len: int = field(default_factory=lambda: 20)

    _gis_cache: Dict[str, np.ndarray] = field(default_factory=dict, init=False, repr=False)
    _sample_id_parts: Optional[List[str]] = field(default=None, init=False, repr=False)
    _lmdb_env: Optional[lmdb.Environment] = field(default=None, init=False, repr=False)

    def __del__(self):
        if self._lmdb_env:
            try:
                self._lmdb_env.close()
            except:
                pass

    @classmethod
    def create(cls, sample_id: str, timestamps: List[str], config: Optional['MetConfig'] = None, **kwargs) -> 'MetSample':
        if config is None:
            config = get_config()
        return cls(sample_id=sample_id, timestamps=timestamps, met_config=config, **kwargs)
    
    # ==============================================================================
    # 基础属性
    # ==============================================================================

    def _get_sample_id_parts(self) -> List[str]:
        if self._sample_id_parts is None:
            self._sample_id_parts = self.sample_id.split('_')
        return self._sample_id_parts
    
    @cached_property
    def task_id(self) -> str: return self._get_sample_id_parts()[0]

    @cached_property
    def region_id(self) -> str: return self._get_sample_id_parts()[1]

    @cached_property
    def time_id(self) -> str: return self._get_sample_id_parts()[2]

    @cached_property
    def station_id(self) -> str: return self._get_sample_id_parts()[3]

    @cached_property
    def radar_type(self) -> str: return self._get_sample_id_parts()[4]

    @cached_property
    def batch_id(self) -> str: return self._get_sample_id_parts()[5]

    @cached_property
    def case_id(self) -> str:
        parts = self._get_sample_id_parts()
        return '_'.join(parts[:4])

    @property
    def dataset_folder_name(self) -> str:
        return "TrainSet" if self.is_train else self.test_set

    @property
    def file_date_format(self) -> str: return self.met_config.file_date_format
    
    @property
    def nwp_prefix(self) -> str: return self.met_config.nwp_prefix
    
    @property
    def gis_data_path(self) -> str: return self.met_config.gis_data_path

    @property
    def metadata(self) -> Dict:
        metadata_dict = vars(self).copy()
        metadata_dict.pop('_gis_cache', None)
        metadata_dict.pop('_lmdb_env', None)
        return metadata_dict

    def str_to_datetime(self, time_str: str) -> datetime:
        return datetime.strptime(time_str, self.file_date_format)

    def datetime_to_str(self, datetime_obj: datetime) -> str:
        return datetime_obj.strftime(self.file_date_format)

    def _get_channel_limits(self, channel: Union[MetLabel, MetRadar, MetNwp, MetGis]) -> tuple[float, float]:
        return (
            float(getattr(channel, "min", 0.0)),
            float(getattr(channel, "max", 1.0))
        )

    # ==============================================================================
    # LMDB 读取核心 (解耦与重构)
    # ==============================================================================

    def _get_lmdb_env(self) -> lmdb.Environment:
        if self._lmdb_env is None:
            db_path = os.path.join(self.lmdb_root, f"{self.region_id}.lmdb")
            # 增加 max_dbs=0 避免不必要的开销，设置合理的 map_size (如果只是读可以忽略)
            self._lmdb_env = lmdb.open(
                db_path, readonly=True, lock=False, readahead=False, meminit=False
            )
        return self._lmdb_env

    def _generate_lmdb_key(self, channel: Union[MetLabel, MetRadar, MetNwp], timestamp: str) -> bytes:
        """
        生成数据文件的 Key
        """
        var_name = channel.value
        filename = ""
        parent_dir = channel.parent
        
        if isinstance(channel, MetRadar):
            filename = f"{self.task_id}_RADA_{self.station_id}_{timestamp}_{self.radar_type}_{var_name}.npy"
        elif isinstance(channel, MetLabel):
            filename = f"{self.task_id}_Label_{var_name}_{self.station_id}_{timestamp}.npy"
        elif isinstance(channel, MetNwp):
            dt = self.str_to_datetime(timestamp)
            if dt.minute >= 30:
                dt += timedelta(hours=1)
            nwp_ts = dt.replace(minute=0).strftime(self.file_date_format)
            filename = f"{self.task_id}_{self.nwp_prefix}_{self.station_id}_{nwp_ts}_{var_name}.npy"

        path_parts = [
            self.task_id, self.dataset_folder_name, self.region_id, self.case_id,
            parent_dir, var_name, filename
        ]
        return "/".join(path_parts).encode('ascii')

    def _generate_lmdb_mask_key(self, timestamp: str) -> bytes:
        """
        [辅助] 生成 Mask Key (仅针对 RA)
        """
        var_name = "MASK"
        parent_dir = "MASK"
        filename = f"{self.task_id}_Label_{var_name}_{self.station_id}_{timestamp}.npy"
        
        path_parts = [
            self.task_id, self.dataset_folder_name, self.region_id, self.case_id,
            parent_dir, var_name, filename
        ]
        return "/".join(path_parts).encode('ascii')

    def _read_mask_from_lmdb(self, timestamp: str) -> np.ndarray:
        """
        [独立] 读取物理 Mask 文件
        返回: 
            - 成功: 解压后的 Mask 数组
            - 失败: 全 False (Zero) 数组，表示该时刻无效
        """
        try:
            env = self._get_lmdb_env()
            key = self._generate_lmdb_mask_key(timestamp)
            
            with env.begin(write=False) as txn:
                buf = txn.get(key)
                if buf:
                    return pickle.loads(zlib.decompress(buf))
        except Exception:
            pass
            
        # 兜底：返回全 0 (False) 数组
        return np.zeros(self.default_shape, dtype=bool)

    def _read_from_lmdb(self, channel: Union[MetLabel, MetRadar, MetNwp], timestamp: str) -> Optional[np.ndarray]:
        """
        [独立] 读取数据文件
        返回: 
            - 成功: 解压后的 Data 数组
            - 失败: None (由上层处理兜底)
        """
        try:
            env = self._get_lmdb_env()
            key = self._generate_lmdb_key(channel, timestamp)
            
            with env.begin(write=False) as txn:
                buf = txn.get(key)
                if buf is None:
                    return None
                return pickle.loads(zlib.decompress(buf))
        except Exception:
            return None

    # ==============================================================================
    # 加载与归一化
    # ==============================================================================

    def _normalize_data(self, data: np.ndarray, channel: Union[MetLabel, MetRadar, MetNwp]) -> np.ndarray:
        min_val, max_val = self._get_channel_limits(channel)
        data = data.astype(np.float32)
        denom = max_val - min_val
        if abs(denom) > 1e-6:
            inv_denom = 1.0 / denom
            data = (data - min_val) * inv_denom
        else:
            data[:] = 0.0
        np.clip(data, 0.0, 1.0, out=data)
        return data

    def _load_mask_frame(self, timestamp: str) -> np.ndarray:
        """
        [核心] 获取当前时间戳的全局 Mask
        """
        return self._read_mask_from_lmdb(timestamp)

    def _load_label_frame(self, var: MetLabel, timestamp: str) -> np.ndarray:
        """
        加载 Label 数据 (不返回 Mask)
        """
        data = self._read_from_lmdb(var, timestamp)
        if data is None: 
            return np.zeros(self.default_shape, dtype=np.float32)
        return self._normalize_data(data, var)

    def _load_radar_frame(self, var: MetRadar, timestamp: str) -> np.ndarray:
        """
        加载 Radar 数据 (不返回 Mask)
        """
        data = self._read_from_lmdb(var, timestamp)
        if data is None: 
            return np.zeros(self.default_shape, dtype=np.float32)
        return self._normalize_data(data, var)

    def _load_nwp_frame(self, var: MetNwp, timestamp: str) -> np.ndarray: 
        """
        加载 NWP 数据 (不返回 Mask)
        """
        data = self._read_from_lmdb(var, timestamp)
        if data is None: 
            return np.zeros(self.default_shape, dtype=np.float32)
        return self._normalize_data(data, var)

    def _load_gis_frame(self, var: MetGis, timestamp: str) -> np.ndarray:
        """
        加载 GIS 数据 (不返回 Mask)
        """
        # 1. 动态计算 Time Embeddings
        if var in [MetGis.MONTH_SIN, MetGis.MONTH_COS, MetGis.HOUR_SIN, MetGis.HOUR_COS]:
            try:
                obs_time = self.str_to_datetime(timestamp)
                raw_val = 0.0
                if var in [MetGis.MONTH_SIN, MetGis.MONTH_COS]:
                    angle = 2 * math.pi * float(obs_time.month) / 12.0
                    raw_val = math.sin(angle) if var == MetGis.MONTH_SIN else math.cos(angle)
                elif var in [MetGis.HOUR_SIN, MetGis.HOUR_COS]:
                    angle = 2 * math.pi * float(obs_time.hour) / 24.0
                    raw_val = math.sin(angle) if var == MetGis.HOUR_SIN else math.cos(angle)
                
                min_val, max_val = self._get_channel_limits(var)
                norm_val = (raw_val - min_val) / (max_val - min_val) if (max_val - min_val) > 0 else 0.5
                norm_val = max(0.0, min(1.0, norm_val))
                return np.full(self.default_shape, norm_val, dtype=np.float32)
            except:
                pass
        
        # 2. 静态 GIS 文件读取
        try:
            file_name = f"{var.value}.npy"
            file_path = os.path.join(self.gis_data_path, self.station_id, file_name)
            if os.path.exists(file_path):
                raw_data = np.load(file_path)
                if raw_data.shape != self.default_shape:
                    raw_data = cv2.resize(raw_data, (self.default_shape[1], self.default_shape[0]), interpolation=cv2.INTER_LINEAR)
                min_val, max_val = self._get_channel_limits(var)
                data = np.nan_to_num(raw_data, nan=min_val)
                denom = max_val - min_val
                if abs(denom) > 1e-6:
                    scale = (data - min_val) / denom
                else:
                    scale = np.zeros_like(data)
                np.clip(scale, 0.0, 1.0, out=scale)
                return scale.astype(np.float32)
        except Exception:
            pass
        return np.zeros(self.default_shape, dtype=np.float32)

    def _preload_gis_data(self):
        self._gis_cache.clear()
        default_data = np.zeros(self.default_shape, dtype=np.float32)
        first_ts = self.timestamps[0]
        
        for channel in self.channels:
            if isinstance(channel, MetGis):
                cache_key = f"gis_{channel.value}"
                data = self._load_gis_frame(channel, first_ts)
                self._gis_cache[cache_key] = data if data is not None else default_data

    def _load_channel_frame_with_fallback(self, channel, timestamp: str, timestep_idx: int, all_timestamps: List[str]) -> np.ndarray:
        """
        [修改] 只返回数据 array
        """
        if isinstance(channel, MetGis):
            cache_key = f"gis_{channel.value}"
            if cache_key in self._gis_cache:
                return self._gis_cache[cache_key]
            return self._load_gis_frame(channel, timestamp)

        # 尝试直接读取
        if isinstance(channel, MetLabel):
            d = self._load_label_frame(channel, timestamp)
        elif isinstance(channel, MetRadar):
            d = self._load_radar_frame(channel, timestamp)
        elif isinstance(channel, MetNwp):
            d = self._load_nwp_frame(channel, timestamp)
        else:
            d = np.zeros(self.default_shape, dtype=np.float32)

        # 这里实际上 _load_..._frame 已经做了 zeros 兜底，但如果是逻辑上的 None 检查可以保留
        # 简单起见，假设返回的 d 总是有效的 ndarray (即便全0)
        
        # 简单的时序填补 (Time Interpolation/Padding) - 只有当数据真的完全缺失时才需要
        # 但鉴于现在 _load_X_frame 返回全0，很难区分是"读到了0"还是"文件不存在"
        # 如果需要严格的时序填补，_load_X_frame 应该返回 None，然后在这里处理
        # 既然采用了 "Zero Padding for Missing"，直接返回 d 即可
        return d

    # ==============================================================================
    # 核心加载接口 (obs_seq_len / pred_seq_len)
    # ==============================================================================

    def load_input_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        加载输入数据 (观测阶段 + 未来引导)
        """
        past_timesteps = self.timestamps[:self.obs_seq_len]
        
        future_start = self.obs_seq_len
        future_end = future_start + self.pred_seq_len
        future_timesteps = self.timestamps[future_start:future_end]
        
        self._preload_gis_data()

        # A. 基础观测 (Observation)
        past_series, past_masks = [], []
        
        for i, ts in enumerate(past_timesteps):
            # 1. 获取该时刻的全局 Mask
            ts_mask = self._load_mask_frame(ts)
            
            # 2. 获取所有通道数据
            frames = []
            for channel in self.channels:
                d = self._load_channel_frame_with_fallback(channel, ts, i, self.timestamps)
                frames.append(d)
            
            # 3. 堆叠
            past_series.append(np.stack(frames, axis=0)) # (C, H, W)
            
            # 4. Mask 扩展: (C, H, W) - 所有通道共享同一个 Mask
            # 广播 ts_mask 到所有通道
            mask_expanded = np.repeat(ts_mask[np.newaxis, :, :], len(self.channels), axis=0)
            past_masks.append(mask_expanded)
            
        input_base = np.stack(past_series, axis=0) # (obs_seq_len, C, H, W)
        mask_base = np.stack(past_masks, axis=0)

        # B. 未来NWP引导 (Future Guidance)
        nwp_channels = [c for c in self.channels if isinstance(c, MetNwp)]
        
        if nwp_channels and len(future_timesteps) == self.pred_seq_len:
            future_series, future_masks = [], []
            for i, ts in enumerate(future_timesteps):
                # 获取未来时刻 Mask
                ts_mask = self._load_mask_frame(ts)
                
                frames = []
                for channel in nwp_channels:
                    # 注意：复用 fallback 逻辑，或者直接调用 load_nwp
                    d = self._load_channel_frame_with_fallback(channel, ts, future_start + i, self.timestamps)
                    frames.append(d)
                
                future_series.append(np.stack(frames, axis=0))
                
                # Mask 扩展
                mask_expanded = np.repeat(ts_mask[np.newaxis, :, :], len(nwp_channels), axis=0)
                future_masks.append(mask_expanded)
            
            input_future = np.stack(future_series, axis=0)
            mask_future = np.stack(future_masks, axis=0)
            
            # Temporal Folding
            B, C, H, W = input_future.shape
            fold_factor = self.pred_seq_len // self.obs_seq_len
            
            if self.pred_seq_len % self.obs_seq_len == 0:
                input_folded = input_future.reshape(self.obs_seq_len, fold_factor * C, H, W)
                mask_folded = mask_future.reshape(self.obs_seq_len, fold_factor * C, H, W)
                
                input_data = np.concatenate([input_base, input_folded], axis=1)
                input_mask = np.concatenate([mask_base, mask_folded], axis=1)
            else:
                input_data, input_mask = input_base, mask_base
        else:
            input_data, input_mask = input_base, mask_base

        return input_data, input_mask

    def load_target_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        加载目标数据 (预测阶段)
        """
        target_data, valid_mask = [], []
        target_timestamps = self.timestamps[self.obs_seq_len : self.obs_seq_len + self.pred_seq_len]

        for ts in target_timestamps:
            # 1. 获取全局 Mask
            ts_mask = self._load_mask_frame(ts)
            
            # 2. 获取 Label 数据
            d = self._load_label_frame(MetLabel.RA, ts)
            
            target_data.append(d)
            valid_mask.append(ts_mask)

        return (
            np.expand_dims(np.stack(target_data, axis=0), axis=1),
            np.expand_dims(np.stack(valid_mask, axis=0), axis=1)
        )

    def to_numpy(self, is_train: bool=False) -> Tuple[Dict, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        input_data, input_mask = self.load_input_data()
        
        if is_train:
            target_data, target_mask = self.load_target_data()
            return self.metadata, input_data, target_data, target_mask, input_mask
        else:
            return self.metadata, input_data, None, None, input_mask