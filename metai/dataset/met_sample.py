# metai/dataset/met_sample.py

import os
import math
import lmdb
import zlib
import pickle
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Union, Tuple
from datetime import datetime, timedelta

# 引入项目配置与枚举
from metai.utils import get_config, MetConfig, MetLabel, MetRadar, MetNwp, MetGis

# ==============================================================================
# 全局常量配置
# ==============================================================================

TARGET_SIZE = 256  # 目标图像尺寸 (H, W)

# 默认基础通道配置 (Base Channels)
# 用于构建 Input Tensor 的前半部分 (当前时刻观测)
# 总数: 1(Label) + 5(Radar) + 6(NWP) + 7(GIS) = 19
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
    气象多模态样本加载类 (Consumer)。
    
    负责管理单个样本的生命周期，包括：
    1. 解析 Sample ID 获取元数据 (时间、地点、雷达类型等)。
    2. 从 LMDB 高效读取稀疏的历史观测数据 (Radar, Label, NWP)。
    3. 从磁盘读取静态 GIS 数据或计算动态时间嵌入。
    4. 执行未来时刻 NWP 数据的读取与时间维度折叠 (Folding)。
    """
    
    # --- 核心参数 ---
    sample_id: str
    timestamps: List[str]
    
    # 配置对象: 若为 None，将在 __post_init__ 中自动加载全局配置
    met_config: Optional[MetConfig] = None 
    
    # --- 数据集参数 ---
    is_train: bool = True
    test_set_name: str = "TestSet"
    
    # LMDB 根路径: 若为 None，优先读取 Config 配置
    lmdb_root: Optional[str] = None 
    
    # --- 调试参数 ---
    verbose: bool = False  # 开启后打印详细的 Key 缺失警告
    
    # --- 序列参数 ---
    obs_seq_len: int = 10  # 观测长度
    pred_seq_len: int = 20 # 预测长度
    
    # --- 通道参数 ---
    channels: List[Union[MetLabel, MetRadar, MetNwp, MetGis]] = field(
        default_factory=lambda: list(_DEFAULT_CHANNELS)
    )
    
    # --- 内部状态 (无需手动初始化) ---
    _lmdb_env: Optional[lmdb.Environment] = field(default=None, init=False, repr=False)
    _sample_parts: Dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _gis_cache: Dict[str, np.ndarray] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        """初始化后处理：配置自动加载与 ID 解析"""
        
        # 1. 自动加载全局配置
        if self.met_config is None:
            self.met_config = get_config()
            if self.verbose:
                print(f"[MetSample] Auto-loaded Config. Prefix={self.nwp_prefix}")

        # 2. 补全 LMDB 路径 (优先级: 参数传入 > Config配置 > 硬编码默认)
        if self.lmdb_root is None:
            self.lmdb_root = getattr(self.met_config, 'lmdb_root_path', "/data/zjobs/LMDB")

        # 3. 解析 Sample ID
        # 格式: TaskID_RegionID_CaseTime_StationID_RadarType_BatchID
        parts = self.sample_id.split('_')
        if len(parts) < 6 and self.verbose:
            print(f"Warning: Invalid Sample ID format: {self.sample_id}")
        
        def get_part(idx): return parts[idx] if len(parts) > idx else "Unknown"
        
        self._sample_parts = {
            'task_id':    get_part(0),
            'region_id':  get_part(1),
            'case_time':  get_part(2),
            'station_id': get_part(3),
            'radar_type': get_part(4),
            'batch_id':   get_part(5),
            'case_id':    '_'.join(parts[:4])
        }

    def __del__(self):
        """析构时尝试释放 LMDB 环境句柄"""
        if self._lmdb_env:
            try:
                self._lmdb_env.close()
            except Exception:
                pass

    # ==============================================================================
    # 属性访问器 (Properties)
    # ==============================================================================

    # --- 配置相关 ---
    @property
    def date_fmt(self) -> str:
        """日期格式 (e.g. %Y%m%d-%H%M)"""
        return getattr(self.met_config, 'file_date_format', '%m%d-%H%M')

    @property
    def nwp_prefix(self) -> str:
        """NWP 文件前缀 (e.g. ERA5, RRA)"""
        return getattr(self.met_config, 'nwp_prefix', 'RRA')

    @property
    def gis_data_path(self) -> str:
        """静态 GIS 文件根目录"""
        return getattr(self.met_config, 'gis_data_path', '/data/zjobs/GIS')

    @property
    def dataset_dir_name(self) -> str:
        """根据模式返回数据集文件夹名称"""
        return "TrainSet" if self.is_train else self.test_set_name

    # --- Sample ID 拆解属性 ---
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

    # --- 通道逻辑 ---
    @property
    def nwp_forcing_channels(self) -> List:
        """
        筛选出用于未来引导 (Forcing) 的通道。
        仅保留 MetNwp 类型，移除 GIS 以避免冗余。
        """
        return [c for c in self.channels if isinstance(c, MetNwp)]

    # ==============================================================================
    # Key 生成逻辑 (必须与 create_lmdb.py 严格一致)
    # ==============================================================================

    def _get_nwp_timestamp_str(self, ts_raw: str) -> str:
        """NWP 时间对齐逻辑: 分钟>=30 进位到下一小时，且分钟置零"""
        try:
            dt = datetime.strptime(ts_raw, self.date_fmt)
            if dt.minute >= 30:
                dt += timedelta(hours=1)
            return dt.replace(minute=0).strftime(self.date_fmt)
        except Exception:
            return ts_raw

    def _construct_lmdb_key(self, channel, ts_raw: str) -> bytes:
        """
        构造 LMDB 键路径。
        格式: Task/Dataset/Region/Case/ParentDir/VarName/Filename
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
        
        key_parts = [
            meta['task_id'], self.dataset_dir_name, meta['region_id'], 
            meta['case_id'], parent_dir, var_name, filename
        ]
        return "/".join(key_parts).encode('ascii')

    def _construct_mask_key(self, ts_raw: str) -> bytes:
        """构造 Mask 键 (基于 RA 键的字符串替换)"""
        ra_key = self._construct_lmdb_key(MetLabel.RA, ts_raw)
        # 假设 Mask 文件命名逻辑仅仅是将变量名 RA 替换为 MASK
        return ra_key.decode('ascii').replace("RA", "MASK").encode('ascii')

    # ==============================================================================
    # 基础 I/O 与 数据处理
    # ==============================================================================

    def _get_env(self) -> lmdb.Environment:
        """获取 LMDB 只读环境句柄 (Singleton)"""
        if self._lmdb_env is None:
            path = os.path.join(self.lmdb_root, f"{self.region_id}.lmdb")
            if self.verbose:
                print(f"[MetSample] Opening LMDB: {path}")
            
            # readonly=True, lock=False 对并发性能至关重要
            self._lmdb_env = lmdb.open(
                path, readonly=True, lock=False, readahead=False, meminit=False
            )
        return self._lmdb_env

    def _read_from_lmdb(self, txn: lmdb.Transaction, key: bytes) -> Optional[np.ndarray]:
        """
        从 LMDB 读取并反序列化数据。
        
        Args:
            txn: 必须提供有效的 LMDB 事务句柄
            key: 数据键值
        """
        try:
            buf = txn.get(key)
            if buf is None:
                if self.verbose:
                    print(f"❌ [LMDB MISSING] {key.decode('ascii')}")
                return None
            return pickle.loads(zlib.decompress(buf))
        except Exception as e:
            if self.verbose:
                print(f"❌ [LMDB ERROR] Reading {key.decode('ascii')}: {e}")
            return None

    def _normalize(self, data: np.ndarray, channel) -> np.ndarray:
        """执行 Min-Max 归一化"""
        min_v = getattr(channel, 'min', 0.0)
        max_v = getattr(channel, 'max', 1.0)
        
        data = data.astype(np.float32)
        denom = max_v - min_v
        
        if abs(denom) > 1e-9:
            data = (data - min_v) / denom
        else:
            data[:] = 0.0 # 避免除以零
            
        return np.clip(data, 0.0, 1.0)

    def _load_gis(self, channel: MetGis, ts: str) -> np.ndarray:
        """
        加载 GIS 数据 (带内存缓存)。
        
        支持：
        1. 动态计算的时间嵌入 (Time Embeddings: SIN/COS)
        2. 静态 NPY 文件读取 (DEM/LAT/LON)，支持 Resize
        """
        cache_key = f"{channel.name}_{ts}"
        if cache_key in self._gis_cache: 
            return self._gis_cache[cache_key]

        data = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.float32)
        
        try:
            # --- Case A: 动态时间嵌入 ---
            if channel in [MetGis.MONTH_SIN, MetGis.MONTH_COS, MetGis.HOUR_SIN, MetGis.HOUR_COS]:
                dt = datetime.strptime(ts, self.date_fmt)
                val = 0.0
                if channel == MetGis.MONTH_SIN: val = math.sin(2 * math.pi * dt.month / 12.0)
                elif channel == MetGis.MONTH_COS: val = math.cos(2 * math.pi * dt.month / 12.0)
                elif channel == MetGis.HOUR_SIN: val = math.sin(2 * math.pi * dt.hour / 24.0)
                elif channel == MetGis.HOUR_COS: val = math.cos(2 * math.pi * dt.hour / 24.0)
                data.fill(val)

            # --- Case B: 静态文件读取 (Lat/Lon/Dem) ---
            elif channel in [MetGis.LAT, MetGis.LON, MetGis.DEM]:
                # 文件名转小写以匹配文件系统 (例如 lat.npy)
                filename = f"{channel.name.lower()}.npy"
                file_path = os.path.join(self.gis_data_path, self.station_id, filename)
                
                if os.path.exists(file_path):
                    raw = np.load(file_path)
                    # 尺寸对齐 (防止原始数据非 256x256)
                    if raw.shape != (TARGET_SIZE, TARGET_SIZE):
                        raw = cv2.resize(raw, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_LINEAR)
                    data = raw
                else:
                    if self.verbose:
                        print(f"❌ [GIS MISSING] {file_path}")

        except Exception as e:
            if self.verbose:
                print(f"❌ [GIS ERROR] Loading {channel.name}: {e}")
            pass
        
        # 归一化处理
        data = self._normalize(data, channel)
        self._gis_cache[cache_key] = data
        return data

    # ==============================================================================
    # 核心逻辑：未来数据融合 (Folding)
    # ==============================================================================

    def _load_future_forcing(self, txn: lmdb.Transaction, pred_ts: List[str]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        加载未来 NWP 数据并折叠时间维度 (Channel Folding)。
        
        转换逻辑: 
        Input Shape:  (T_pred, C_nwp, H, W)
        Output Shape: (T_obs, Ratio*C_nwp, H, W) 
        
        例如: T_pred=20, T_obs=10, C_nwp=6 -> Ratio=2 -> Output Channels=12
        """
        # 仅加载 NWP 通道
        forcing_channels = self.nwp_forcing_channels
        if not forcing_channels:
            return None, None
        
        raw_list, mask_list = [], []
        
        for ts in pred_ts:
            # 1. 借用该时刻的 Label Mask 确定有效性 (通常 NWP 没有独立的 Mask 文件)
            m_key = self._construct_mask_key(ts)
            m_buf = self._read_from_lmdb(txn, m_key)
            valid_mask = np.ones((TARGET_SIZE, TARGET_SIZE), bool) if m_buf is None else m_buf.astype(bool)
            
            # 2. 读取所有 NWP 通道
            frame_data = []
            for ch in forcing_channels:
                key = self._construct_lmdb_key(ch, ts)
                raw = self._read_from_lmdb(txn, key)
                d = np.zeros((TARGET_SIZE, TARGET_SIZE), np.float32) if raw is None else self._normalize(raw, ch)
                frame_data.append(d)
            
            raw_list.append(np.stack(frame_data, axis=0)) 
            mask_list.append(valid_mask[np.newaxis, ...])

        future_tensor = np.stack(raw_list, axis=0) # (T_pred, C_nwp, H, W)
        future_mask = np.stack(mask_list, axis=0)

        # 3. 执行折叠 (Folding)
        T_pred = len(pred_ts)
        T_obs = self.obs_seq_len
        
        # 仅当预测长度能被观测长度整除时执行
        if T_pred > 0 and T_pred % T_obs == 0:
            ratio = T_pred // T_obs
            C, H, W = future_tensor.shape[1:]
            
            # 重塑: (T_obs, Ratio, C, H, W)
            folded = future_tensor.reshape(T_obs, ratio, C, H, W)
            mask_folded = future_mask.reshape(T_obs, ratio, 1, H, W)
            
            # 合并通道: (T_obs, Ratio*C, H, W)
            folded = folded.reshape(T_obs, ratio * C, H, W)
            
            # Mask 合并: 在时间维度取交集 (只要任意一个折叠步无效，则整体无效 -> Min/Logical And)
            mask_final = mask_folded.min(axis=1) 
            
            return folded, mask_final
        
        return None, None

    # ==============================================================================
    # 外部调用接口 (Public API)
    # ==============================================================================

    def to_numpy(self) -> Tuple[Dict, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        加载并处理完整的样本数据。
        
        关键优化:
        在此处开启单次 LMDB Transaction，并传递给所有子函数使用。
        这比在每个 read 函数中反复开启 Transaction 快数倍。
        
        Returns:
            meta (Dict): 样本元数据
            input_tensor (ndarray): (T=10, C=31, H, W) [19 Base + 12 Folded]
            target_tensor (ndarray): (T=20, C=1, H, W) [Rain] 或 None
            input_mask (ndarray): (T=10, 1, H, W)
            target_mask (ndarray): (T=20, 1, H, W) 或 None
        """
        env = self._get_env()
        
        # 切分时间序列
        obs_ts = self.timestamps[:self.obs_seq_len]
        pred_ts = self.timestamps[self.obs_seq_len : self.obs_seq_len + self.pred_seq_len]
        
        input_tensor, input_mask = None, None
        target_tensor, target_mask = None, None
        
        # 开启只读事务 (Context Manager 自动处理 commit/abort)
        with env.begin(write=False) as txn:
            
            # --- Part A: 过去观测 (Observation) ---
            # 包含: RA(1) + Radar(5) + NWP(6) + GIS(7) = 19 通道
            obs_list, obs_mask_list = [], []
            for ts in obs_ts:
                # Mask 读取
                m_buf = self._read_from_lmdb(txn, self._construct_mask_key(ts))
                valid_mask = np.zeros((TARGET_SIZE, TARGET_SIZE), bool) if m_buf is None else m_buf.astype(bool)
                
                # Data Channels 读取
                frame = []
                for ch in self.channels:
                    if isinstance(ch, MetGis):
                        d = self._load_gis(ch, ts) # 从磁盘或计算读取，不需要 txn
                    else:
                        key = self._construct_lmdb_key(ch, ts)
                        raw = self._read_from_lmdb(txn, key) # 传入 txn
                        d = np.zeros((TARGET_SIZE, TARGET_SIZE), np.float32) if raw is None else self._normalize(raw, ch)
                    frame.append(d)
                
                obs_list.append(np.stack(frame, axis=0))
                obs_mask_list.append(valid_mask[np.newaxis, ...])
            
            base_input = np.stack(obs_list, axis=0) # (T_obs, C_base, H, W)
            base_mask = np.stack(obs_mask_list, axis=0)

            # --- Part B: 未来 NWP 折叠 (Forcing) ---
            # 包含: NWP(6) * 2 = 12 通道 (假设 Ratio=2)
            folded_nwp, folded_mask = self._load_future_forcing(txn, pred_ts)
            
            if folded_nwp is not None:
                # 拼接: 19 + 12 = 31 通道
                input_tensor = np.concatenate([base_input, folded_nwp], axis=1)
                input_mask = base_mask & folded_mask 
            else:
                input_tensor = base_input
                input_mask = base_mask

            # --- Part C: 预测目标 (Target) ---
            if self.is_train:
                t_list, tm_list = [], []
                for ts in pred_ts:
                    # 读取降水 (RA)
                    ra_key = self._construct_lmdb_key(MetLabel.RA, ts)
                    raw = self._read_from_lmdb(txn, ra_key)
                    
                    # 读取 Mask
                    m_key = self._construct_mask_key(ts)
                    m_buf = self._read_from_lmdb(txn, m_key)
                    
                    if raw is not None and m_buf is not None:
                        d = self._normalize(raw, MetLabel.RA)
                        m = m_buf.astype(bool)
                    else:
                        d = np.zeros((TARGET_SIZE, TARGET_SIZE), np.float32)
                        m = np.zeros((TARGET_SIZE, TARGET_SIZE), bool)
                    
                    t_list.append(d[np.newaxis, ...])
                    tm_list.append(m[np.newaxis, ...])
                
                target_tensor = np.stack(t_list, axis=0)
                target_mask = np.stack(tm_list, axis=0)

        # 构建元数据返回
        meta_out = self._sample_parts.copy()
        meta_out['sample_id'] = self.sample_id
        meta_out['timestamps'] = self.timestamps
        
        # 返回值顺序: meta, input, target, input_mask, target_mask
        return meta_out, input_tensor, target_tensor, input_mask, target_mask