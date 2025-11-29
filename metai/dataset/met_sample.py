# metai/dataset/met_sample.py

import os
import math
import lmdb
import zlib
import pickle
import cv2
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import List, NamedTuple, Optional, Dict, Union, Tuple, Any
from datetime import datetime, timedelta

# 引入项目配置与枚举
from metai.utils import get_config, MetConfig, MetLabel, MetRadar, MetNwp, MetGis

# ==============================================================================
# 全局常量与类型定义
# ==============================================================================

TARGET_SIZE = 256  # 目标图像尺寸 (H, W)
logger = logging.getLogger(__name__)

# 类型别名：整合所有可能的通道类型
ChannelType = Union[MetLabel, MetRadar, MetNwp, MetGis]

# 默认基础通道配置
_DEFAULT_CHANNELS: List[ChannelType] = [
    MetLabel.RA, 
    MetRadar.CR, MetRadar.CAP30, MetRadar.CAP50, MetRadar.ET, MetRadar.VIL,
    MetNwp.WS925, MetNwp.WS500, MetNwp.Q850, MetNwp.Q700, MetNwp.PWAT, MetNwp.CAPE,
    MetGis.LAT, MetGis.LON, MetGis.DEM, 
    MetGis.MONTH_SIN, MetGis.MONTH_COS, MetGis.HOUR_SIN, MetGis.HOUR_COS
]

class BatchData(NamedTuple):
    """
    单个样本的数据容器 (NamedTuple)。
    
    Attributes:
        meta (Dict[str, Any]): 样本元数据（ID、时间戳、站点信息等）。
        x (np.ndarray): 输入张量。Shape: (T_obs, C_in, H, W)。
        y (Optional[np.ndarray]): 目标张量（仅训练模式）。Shape: (T_pred, C_out, H, W)。
        x_mask (np.ndarray): 输入数据的有效性掩码。Shape: (T_obs, 1, H, W)。
        y_mask (Optional[np.ndarray]): 目标数据的有效性掩码。Shape: (T_pred, 1, H, W)。
    """
    meta: Dict[str, Any]
    x: np.ndarray
    y: Optional[np.ndarray]
    x_mask: np.ndarray
    y_mask: Optional[np.ndarray]

@dataclass
class MetSample:
    """
    气象多模态样本加载器 (MetSample Consumer)。
    
    功能：
    1. 解析 Sample ID 获取元数据。
    2. 从 LMDB 高效读取稀疏历史观测 (Radar/NWP)。
    3. 计算或读取静态 GIS 嵌入。
    4. 执行未来 NWP 数据的读取与时间维度折叠 (Folding)。

    Attributes:
        sample_id (str): 唯一样本标识符。
        timestamps (List[str]): 完整的时间戳序列 (历史 + 未来)。
        met_config (Optional[MetConfig]): 全局配置对象。
        is_train (bool): 是否为训练模式（决定是否加载 Label）。
        lmdb_root (Optional[str]): LMDB 数据库根目录。
        obs_seq_len (int): 过去观测帧数 (Input)。
        pred_seq_len (int): 未来预测帧数 (Target)。
        channels (List[ChannelType]): 需要加载的通道列表。
    """
    
    # --- 核心参数 ---
    sample_id: str
    timestamps: List[str]
    
    # --- 配置参数 ---
    met_config: Optional[MetConfig] = None 
    is_train: bool = True
    testset_name: str = "TestSet"
    lmdb_root: Optional[str] = None 
    
    # --- 调试参数 ---
    verbose: bool = False  # 是否打印详细的 Key 缺失日志
    
    # --- 序列参数 ---
    obs_seq_len: int = 10 
    pred_seq_len: int = 20 
    
    # --- 通道参数 ---
    channels: List[ChannelType] = field(default_factory=lambda: list(_DEFAULT_CHANNELS))
    
    # --- 内部状态 (无需手动初始化) ---
    _lmdb_env: Optional[lmdb.Environment] = field(default=None, init=False, repr=False)
    _sample_parts: Dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _gis_cache: Dict[str, np.ndarray] = field(default_factory=dict, init=False, repr=False)
    _external_env: bool = field(default=False, init=False, repr=False) # 标记是否使用了外部传入的 env

    def __post_init__(self):
        """初始化后处理：加载配置与解析 ID。"""
        # 1. 加载全局配置
        if self.met_config is None:
            self.met_config = get_config()

        # 2. 确定 LMDB 根路径
        if self.lmdb_root is None:
            self.lmdb_root = getattr(self.met_config, 'lmdb_root_path', "/data/zjobs/LMDB")

        # 3. 解析 Sample ID
        self._parse_sample_id()

    def _parse_sample_id(self):
        """解析 sample_id 字符串到结构化元数据字典。"""
        parts = self.sample_id.split('_')
        
        # 简单校验
        if len(parts) < 6:
            msg = f"Invalid Sample ID format: {self.sample_id}. Expected >= 6 parts."
            if self.verbose:
                logger.warning(msg)
        
        def safe_get(idx): return parts[idx] if len(parts) > idx else "Unknown"
        
        self._sample_parts = {
            'task_id':    safe_get(0),
            'region_id':  safe_get(1),
            'case_time':  safe_get(2),
            'station_id': safe_get(3),
            'radar_type': safe_get(4),
            'batch_id':   safe_get(5),
            'case_id':    '_'.join(parts[:4])
        }

    # ==============================================================================
    # 资源管理 (Context Manager)
    # ==============================================================================

    def set_env(self, env: lmdb.Environment):
        """
        注入现有的 LMDB 环境句柄。
        
        Args:
            env (lmdb.Environment): 打开的 LMDB 环境。
        """
        self._lmdb_env = env
        self._external_env = True

    def __enter__(self):
        """进入上下文：确保 LMDB 环境已打开。"""
        self._get_env()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文：如果是内部创建的 env，则关闭它以释放文件句柄。"""
        if not self._external_env and self._lmdb_env:
            self._lmdb_env.close()
            self._lmdb_env = None

    # ==============================================================================
    # 属性访问器 (Properties)
    # ==============================================================================

    @property
    def date_fmt(self) -> str: return getattr(self.met_config, 'file_date_format', '%m%d-%H%M')
    
    @property
    def nwp_prefix(self) -> str: return getattr(self.met_config, 'nwp_prefix', 'RRA')
    
    @property
    def gis_data_path(self) -> str: return getattr(self.met_config, 'gis_data_path', '/data/zjobs/GIS')
    
    @property
    def dataset_dir_name(self) -> str: return "TrainSet" if self.is_train else self.testset_name

    # ID 快捷属性
    @property
    def case_id(self) -> str: return self._sample_parts.get('case_id', '')
    @property
    def region_id(self) -> str: return self._sample_parts.get('region_id', '')
    @property
    def task_id(self) -> str: return self._sample_parts.get('task_id', '')
    @property
    def station_id(self) -> str: return self._sample_parts.get('station_id', '')
    @property
    def radar_type(self) -> str: return self._sample_parts.get('radar_type', '')

    @property
    def nwp_forcing_channels(self) -> List[ChannelType]:
        """筛选出用于未来驱动 (Forcing) 的纯 NWP 通道。"""
        return [c for c in self.channels if isinstance(c, MetNwp)]

    # ==============================================================================
    # 核心逻辑：Key 构造
    # ==============================================================================

    def _get_nwp_timestamp_str(self, ts_raw: str) -> str:
        """
        NWP 时间对齐逻辑。
        
        规则：分钟 >= 30 则进位到下一小时，且分钟强制置零（NWP 通常为逐小时数据）。
        """
        try:
            dt = datetime.strptime(ts_raw, self.date_fmt)
            if dt.minute >= 30:
                dt += timedelta(hours=1)
            return dt.replace(minute=0).strftime(self.date_fmt)
        except ValueError:
            return ts_raw

    def _construct_lmdb_key(self, channel: ChannelType, ts_raw: str) -> bytes:
        """
        构造 LMDB 查询键 (Key)。
        
        Key 格式: Task/Dataset/Region/Case/Parent/Var/Filename
        """
        meta = self._sample_parts
        var_name = channel.name
        parent_dir = channel.parent
        
        filename = ""
        if isinstance(channel, MetRadar):
            filename = f"{meta['task_id']}_RADA_{meta['station_id']}_{ts_raw}_{meta['radar_type']}_{var_name}.npy"
        elif isinstance(channel, MetLabel):
            filename = f"{meta['task_id']}_Label_{var_name}_{meta['station_id']}_{ts_raw}.npy"
        elif isinstance(channel, MetNwp):
            nwp_ts = self._get_nwp_timestamp_str(ts_raw)
            filename = f"{meta['task_id']}_{self.nwp_prefix}_{meta['station_id']}_{nwp_ts}_{var_name}.npy"
        
        key_str = f"{meta['task_id']}/{self.dataset_dir_name}/{meta['region_id']}/{meta['case_id']}/{parent_dir}/{var_name}/{filename}"
        return key_str.encode('ascii')

    def _construct_mask_key(self, ts_raw: str) -> bytes:
        """基于 RA 的键构造 Mask 键 (文件名替换 RA -> MASK)。"""
        ra_key = self._construct_lmdb_key(MetLabel.RA, ts_raw)
        return ra_key.replace(b"RA", b"MASK")

    # ==============================================================================
    # I/O 操作封装
    # ==============================================================================

    def _get_env(self) -> lmdb.Environment:
        """惰性加载 LMDB 环境。"""
        if self._lmdb_env is None:
            db_path = os.path.join(self.lmdb_root, f"{self.region_id}.lmdb")
            if self.verbose:
                print(f"[MetSample] Opening LMDB: {db_path}")
            
            # lock=False 对于只读并发读取至关重要，能避免锁竞争
            self._lmdb_env = lmdb.open(
                db_path, readonly=True, lock=False, readahead=False, meminit=False
            )
        return self._lmdb_env

    def _read_from_lmdb(self, txn: lmdb.Transaction, key: bytes) -> Optional[np.ndarray]:
        """
        从 LMDB 读取并反序列化 (Pickle + Zlib)。
        
        Returns:
            np.ndarray: 数据数组，如果 Key 不存在或出错则返回 None。
        """
        try:
            buf = txn.get(key)
            if buf is None:
                if self.verbose: 
                    print(f"❌ [MISSING] {key.decode('ascii', errors='ignore')}")
                return None
            return pickle.loads(zlib.decompress(buf))
        except Exception as e:
            if self.verbose:
                print(f"❌ [ERROR] {key.decode('ascii', errors='ignore')}: {e}")
            return None

    def _normalize(self, data: np.ndarray, channel: ChannelType) -> np.ndarray:
        """执行 Min-Max 归一化，并转为 float32。"""
        min_v = getattr(channel, 'min', 0.0)
        max_v = getattr(channel, 'max', 1.0)
        
        data = data.astype(np.float32)
        denom = max_v - min_v
        
        if abs(denom) > 1e-9:
            data = (data - min_v) / denom
        else:
            data[:] = 0.0
            
        return np.clip(data, 0.0, 1.0)

    def _load_gis(self, channel: MetGis, ts: str) -> np.ndarray:
        """
        加载 GIS 数据。
        
        包含两种类型：
        1. 动态计算的时间嵌入 (Sin/Cos of Month/Hour)。
        2. 静态地理文件 (DEM, Lat, Lon)。
        """
        # 注意：如果 MetSample 每次都是新建的，这个 cache 仅在单次 Batch 内有效
        cache_key = f"{channel.name}_{ts}"
        if cache_key in self._gis_cache:
            return self._gis_cache[cache_key]

        data = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.float32)
        
        try:
            # 1. 动态时间嵌入
            if channel in [MetGis.MONTH_SIN, MetGis.MONTH_COS, MetGis.HOUR_SIN, MetGis.HOUR_COS]:
                dt = datetime.strptime(ts, self.date_fmt)
                val = 0.0
                if channel == MetGis.MONTH_SIN: val = math.sin(2 * math.pi * dt.month / 12.0)
                elif channel == MetGis.MONTH_COS: val = math.cos(2 * math.pi * dt.month / 12.0)
                elif channel == MetGis.HOUR_SIN: val = math.sin(2 * math.pi * dt.hour / 24.0)
                elif channel == MetGis.HOUR_COS: val = math.cos(2 * math.pi * dt.hour / 24.0)
                data.fill(val)

            # 2. 静态文件 (DEM/LAT/LON)
            elif channel in [MetGis.LAT, MetGis.LON, MetGis.DEM]:
                filename = f"{channel.name.lower()}.npy"
                path = os.path.join(self.gis_data_path, self.station_id, filename)
                
                if os.path.exists(path):
                    raw = np.load(path)
                    # 尺寸安全校验
                    if raw.shape != (TARGET_SIZE, TARGET_SIZE):
                        raw = cv2.resize(raw, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_LINEAR)
                    data = raw
                elif self.verbose:
                    print(f"❌ [GIS MISSING] {path}")

        except Exception as e:
            if self.verbose: print(f"❌ [GIS ERROR] {channel.name}: {e}")

        # 归一化并缓存
        data = self._normalize(data, channel)
        self._gis_cache[cache_key] = data
        return data

    # ==============================================================================
    # 数据加载主逻辑 (模块化)
    # ==============================================================================

    def _fetch_observation(self, txn: lmdb.Transaction, obs_ts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        加载历史观测数据 (Observation)。
        
        Args:
            txn: LMDB 事务句柄。
            obs_ts: 观测时间戳列表。
            
        Returns:
            frames: (T_obs, C, H, W)
            masks: (T_obs, 1, H, W)
        """
        frames, masks = [], []

        for ts in obs_ts:
            # 1. 加载 Mask (如果缺失默认为 False，即无效)
            m_buf = self._read_from_lmdb(txn, self._construct_mask_key(ts))
            valid_mask = np.zeros((TARGET_SIZE, TARGET_SIZE), bool) if m_buf is None else m_buf.astype(bool)
            
            # 2. 加载各通道数据
            ch_data = []
            for ch in self.channels:
                if isinstance(ch, MetGis):
                    d = self._load_gis(ch, ts) 
                else:
                    raw = self._read_from_lmdb(txn, self._construct_lmdb_key(ch, ts))
                    if raw is None:
                        # 关键：如果观测数据缺失，填充 0。是否将 mask 设为 False 取决于业务需求，
                        # 此处保持保守策略，仅填充数据，Mask 仍由 MASK 文件决定。
                        d = np.zeros((TARGET_SIZE, TARGET_SIZE), np.float32)
                    else:
                        d = self._normalize(raw, ch)
                ch_data.append(d)
            
            frames.append(np.stack(ch_data, axis=0))
            masks.append(valid_mask[np.newaxis, ...])

        return np.stack(frames, axis=0), np.stack(masks, axis=0)

    def _fetch_forcing(self, txn: lmdb.Transaction, pred_ts: List[str]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        加载未来 NWP 数据并执行 'Channel Folding' (时间维度折叠)。
        
        变换逻辑: (T_pred, C, H, W) -> (T_obs, C * Ratio, H, W)
        例如: 预测20小时，观测10小时 -> 每2个预测时刻的数据叠在一起，作为1个观测时刻的特征。
        
        Returns:
            folded: (T_obs, C_folded, H, W) 或 None
            mask_final: (T_obs, 1, H, W) 或 None
        """
        forcing_chs = self.nwp_forcing_channels
        if not forcing_chs or not pred_ts:
            return None, None

        # 1. 读取原始未来帧
        raw_list, mask_list = [], []
        for ts in pred_ts:
            # Mask 默认 True (如果文件缺失), 但如果数据本身缺失则会强制修正为 False
            m_buf = self._read_from_lmdb(txn, self._construct_mask_key(ts))
            valid_mask = np.ones((TARGET_SIZE, TARGET_SIZE), bool) if m_buf is None else m_buf.astype(bool)
            
            frame_ch = []
            any_ch_missing = False # 标记当前帧是否有通道缺失

            for ch in forcing_chs:
                raw = self._read_from_lmdb(txn, self._construct_lmdb_key(ch, ts))
                if raw is None:
                    d = np.zeros((TARGET_SIZE, TARGET_SIZE), np.float32)
                    any_ch_missing = True
                else:
                    d = self._normalize(raw, ch)
                frame_ch.append(d)
            
            # 如果 NWP 任何通道缺失，则该帧必须标记为无效，防止模型学习全黑数据
            if any_ch_missing:
                valid_mask[:] = False

            raw_list.append(np.stack(frame_ch, axis=0))
            mask_list.append(valid_mask[np.newaxis, ...])

        future_tensor = np.stack(raw_list, axis=0) # (T_pred, C_nwp, H, W)
        future_mask = np.stack(mask_list, axis=0)

        # 2. 执行时间折叠 (Folding)
        T_pred = len(pred_ts)
        T_obs = self.obs_seq_len
        
        # 校验折叠条件
        if T_pred == 0 or T_pred % T_obs != 0:
            if self.verbose: 
                print(f"[Folding Error] T_pred ({T_pred}) 必须能被 T_obs ({T_obs}) 整除")
            return None, None

        ratio = T_pred // T_obs
        C, H, W = future_tensor.shape[1:]

        # Reshape: (T_obs, Ratio, C, H, W) -> (T_obs, Ratio*C, H, W)
        folded = future_tensor.reshape(T_obs, ratio, C, H, W).reshape(T_obs, ratio * C, H, W)
        
        # Mask 合并: 在折叠的 Ratio 维度上取交集 (Min/Logical AND)
        # 意为: 如果未来 N 小时中任意 1 小时无效，则该折叠步视为无效
        mask_folded = future_mask.reshape(T_obs, ratio, 1, H, W)
        mask_final = mask_folded.min(axis=1) 
        
        return folded, mask_final

    def _fetch_target(self, txn: lmdb.Transaction, pred_ts: List[str]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """加载训练目标 (Ground Truth Rain)。"""
        if not self.is_train:
            return None, None

        t_list, tm_list = [], []
        for ts in pred_ts:
            # 读取 RA
            raw = self._read_from_lmdb(txn, self._construct_lmdb_key(MetLabel.RA, ts))
            # 读取 Mask
            m_buf = self._read_from_lmdb(txn, self._construct_mask_key(ts))
            
            if raw is not None and m_buf is not None:
                d = self._normalize(raw, MetLabel.RA)
                m = m_buf.astype(bool)
            else:
                d = np.zeros((TARGET_SIZE, TARGET_SIZE), np.float32)
                m = np.zeros((TARGET_SIZE, TARGET_SIZE), bool) # 目标缺失即无效
            
            t_list.append(d[np.newaxis, ...])
            tm_list.append(m[np.newaxis, ...])
            
        return np.stack(t_list, axis=0), np.stack(tm_list, axis=0)

    # ==============================================================================
    # 外部调用接口 (API)
    # ==============================================================================

    def to_numpy(self) -> BatchData:
        """
        加载并返回所有张量数据。
        
        Returns:
            BatchData: 包含元数据、输入、目标和掩码的命名元组。
        
        Raises:
            ValueError: 如果提供的时间戳列表长度不足。
        """
        # 校验时间戳长度
        total_len = self.obs_seq_len + self.pred_seq_len
        if len(self.timestamps) < total_len:
            raise ValueError(f"时间戳数量不足: {len(self.timestamps)} < {total_len}")

        obs_ts = self.timestamps[:self.obs_seq_len]
        pred_ts = self.timestamps[self.obs_seq_len : total_len]
        
        env = self._get_env()
        
        # 使用单个事务 (Transaction) 完成所有读取，最大化性能
        with env.begin(write=False) as txn:
            
            # 1. 获取过去观测
            base_input, base_mask = self._fetch_observation(txn, obs_ts)

            # 2. 获取未来 Forcing (NWP Folded)
            folded_nwp, folded_mask = self._fetch_forcing(txn, pred_ts)

            # 3. 拼接 Input (Base + Folded)
            if folded_nwp is not None:
                input_tensor = np.concatenate([base_input, folded_nwp], axis=1)
                input_mask = base_mask & folded_mask
            else:
                input_tensor = base_input
                input_mask = base_mask

            # 4. 获取 Target (Ground Truth)
            target_tensor, target_mask = self._fetch_target(txn, pred_ts)

        # 5. 组装返回数据
        meta_out = self._sample_parts.copy()
        meta_out.update({
            'sample_id': self.sample_id,
            'timestamps': self.timestamps,
            'input_shape': input_tensor.shape
        })
        
        return BatchData(
            meta=meta_out, 
            x=input_tensor, 
            y=target_tensor, 
            x_mask=input_mask, 
            y_mask=target_mask
        )