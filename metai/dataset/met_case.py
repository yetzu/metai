import os
from dataclasses import dataclass
from typing import Optional, List
from datetime import datetime, timedelta
from metai.utils import MLOGE
from metai.utils import MetLabel, MetRadar, MetNwp
from metai.utils.met_config import get_config, MetConfig

@dataclass
class MetCase:
    """
    天气个例数据结构，基于竞赛数据目录结构设计。
    用于管理单个天气个例的元数据和配置信息。
    """
    case_id: str # 个例唯一标识符(如: CP_00_08220804_00093)
    
    # 业务属性
    task_id: str        # 任务类别: CP(短时强降水), TSW(雷暴大风), HA(冰雹)
    region_id: str      # 区域编码 (如: AH, CP)
    station_id: str     # 雷达站点代码 (如: Z9796, 00093)
    base_path: str       # 天气个例数据集根目录路径
    
    is_debug: bool = False  # 是否为调试模式
    radar_type: str = 'SA' # 雷达类型: SA, SB, SC

    # 数据集类型配置
    is_train: bool = True        # 是否为训练集
    test_set: str = "TestSet"      # 测试集类型: "TestA" 或 "TestB"
    
    def __post_init__(self):
        self.label_files = self._load_label_files()
    
    @classmethod
    def create(cls, case_id: str, config: Optional['MetConfig'] = None, **kwargs) -> 'MetCase':
        """
        根据case_id创建MetCase对象
        
        Args:
            case_id: 个例唯一标识符
            config: 配置对象，如果为None则自动获取
        """
        parts = case_id.split('_')
        if len(parts) != 4:
            raise ValueError(f"[ERROR] Invalid case_id format: {case_id}")
        
        task_id, region_id, _, station_id = parts

        # 获取配置，如果未提供则自动获取
        if config is None:
            config = get_config(is_debug=kwargs.get('is_debug', False))
        
        base_path = os.path.join(
            config.root_path,
            task_id,
            "TrainSet" if kwargs.get('is_train', True) else kwargs.get('test_set', "TestSet"),
            region_id,
            case_id,
        )
        
        # 从kwargs中提取is_debug，避免重复传递
        is_debug = kwargs.pop('is_debug', True)
        
        return cls(
            case_id=case_id,
            task_id=task_id,
            region_id=region_id,
            station_id=station_id,
            base_path=base_path,
            is_debug=is_debug,
            **kwargs
        )
    
    def _load_label_files(self, label_var: MetLabel = MetLabel.RA, return_full_path=False) -> List[str]:
        """
        加载当前个例标签文件列表
        
        Returns:
            List[str]: 标签文件路径列表
        """
        
        data_dir = os.path.join(
            self.base_path,
            "LABEL",
            label_var.name
        )
        try:
            if return_full_path:
                return sorted([os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith('.npy')])
            else:
                return sorted([file for file in os.listdir(data_dir) if file.endswith('.npy')])  
        except Exception as e:
            MLOGE(f"加载标签文件失败: {e}")
            return []
    
    def _load_radar_files(self, radar_var: MetRadar = MetRadar.CR, return_full_path=False) -> List[str]:
        """
        加载当前个例雷达文件列表
        
        Args:
            radar_type: 雷达数据类型, 可选: CAP20  CAP30  CAP40  CAP50  CAP60  CAP70  CR  ET  HBR  VIL
        Returns:
            List[str]: 标签文件路径列表
        """
        
        data_dir = os.path.join(
            self.base_path,
            "RADAR",
            radar_var.value
        )
        
        try:
            if return_full_path:
                return sorted([os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith('.npy')])
            else:
                return sorted([file for file in os.listdir(data_dir) if file.endswith('.npy')])
        except Exception as e:
            MLOGE(f"加载雷达文件失败: {e}")
            return []

    def _load_nwp_files(self, nwp_type: MetNwp = MetNwp.CAPE, return_full_path=False) -> List[str]:
        """
        加载当前个例NWP文件列表
        
        Args:
            nwp_type: NWP数据类型, 可选: CAPE  DVG850  HT0  HTw0  LCL  LI300  PE  Q1000  Q850  RH500  TTdMax74  TdSfc850  WS700  muLCL  DVG200  DVG925  HT10  KI  LFC  LI500  PWAT  Q700  RH1000  RH700  TTdMean74  WS500  WS925
        Returns:
            List[str]: NWP文件路径列表
        """
        
        data_dir = os.path.join(
            self.base_path,
            "NWP",
            nwp_type.name
        )

        try:
            if return_full_path:
                return sorted([os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith('.npy')])
            else:
                return sorted([file for file in os.listdir(data_dir) if file.endswith('.npy')])
        except Exception as e:
            MLOGE(f"加载数值预报文件失败: {e}")
            return []

    def _extract_timestamp_from_label_file(self, filename: str) -> Optional[datetime]:
        """
        从标签文件名中提取时间戳
        支持格式：CP_Label_RA_Z9559_20180704-1213.npy 或者 CP_Label_RA_00093_0511-1712.npy
        """
        name_without_ext = filename.replace('.npy', '')
        parts = name_without_ext.split('_')
        if len(parts) >= 4:
            date_time = parts[-1]
            config = get_config()
            
            # 使用配置的日期格式
            date_format = config.get_date_format()
            try:
                return datetime.strptime(date_time, date_format)
            except ValueError:
                return None
        
        return None

    def _validate_label_time_consistency(self, max_interval_minutes: int = 10) -> List[List[str]]:
        """
        检验labels时间一致性，前后文件不应超过指定时间间隔
        
        Args:
            max_interval_minutes: 最大允许的时间间隔（分钟），默认10分钟
            
        Returns:
            List[List[str]]: 如果通过一致性检验则返回 [label_files]，
                           如果不一致则返回多段 [label_files1, label_files2, ...]
        """
        label_files = self._load_label_files(return_full_path=True)
        
        if len(label_files) < 2:
            return [label_files] if label_files else [[]]
        
        # 找到需要拆分的位置
        split_indices = []
        for i in range(len(label_files) - 1):
            timestamp1 = self._extract_timestamp_from_label_file(os.path.basename(label_files[i]))
            timestamp2 = self._extract_timestamp_from_label_file(os.path.basename(label_files[i + 1]))
            
            if timestamp1 and timestamp2:
                interval_minutes = (timestamp2 - timestamp1).total_seconds() / 60
                if interval_minutes > max_interval_minutes:
                    split_indices.append(i + 1)
        
        # 根据拆分位置分割数组
        if not split_indices:
            return [label_files]
        
        split_arrays = []
        start_idx = 0
        
        for split_idx in split_indices:
            split_arrays.append(label_files[start_idx:split_idx])
            start_idx = split_idx
        
        # 添加最后一个数组
        if start_idx < len(label_files):
            split_arrays.append(label_files[start_idx:])
        
        return split_arrays
    
    def _validate_label_time_consistency_for_files(self, label_files: List[str], max_interval_minutes: int = 10) -> List[List[str]]:
        """
        对指定的label文件列表进行时间一致性检验和拆分
        
        Args:
            label_files: 标签文件路径列表
            max_interval_minutes: 最大允许的时间间隔（分钟）
            
        Returns:
            List[List[str]]: 拆分后的文件数组列表
        """
        if len(label_files) < 2:
            return [label_files] if label_files else [[]]
        
        # 按时间戳排序文件，过滤掉无法解析时间戳的文件
        files_with_timestamps = []
        for file_path in label_files:
            timestamp = self._extract_timestamp_from_label_file(os.path.basename(file_path))
            if timestamp:
                files_with_timestamps.append((file_path, timestamp))
        
        # 按时间戳排序
        files_with_timestamps.sort(key=lambda x: x[1])
        sorted_files = [file_path for file_path, _ in files_with_timestamps]
        
        # 找到需要拆分的位置
        split_indices = []
        for i in range(len(sorted_files) - 1):
            timestamp1 = self._extract_timestamp_from_label_file(os.path.basename(sorted_files[i]))
            timestamp2 = self._extract_timestamp_from_label_file(os.path.basename(sorted_files[i + 1]))
            
            if timestamp1 and timestamp2:
                interval_minutes = (timestamp2 - timestamp1).total_seconds() / 60
                if interval_minutes > max_interval_minutes:
                    split_indices.append(i + 1)
        
        # 根据拆分位置分割数组
        if not split_indices:
            return [sorted_files]
        
        split_arrays = []
        start_idx = 0
        
        for split_idx in split_indices:
            split_arrays.append(sorted_files[start_idx:split_idx])
            start_idx = split_idx
        
        # 添加最后一个数组
        if start_idx < len(sorted_files):
            split_arrays.append(sorted_files[start_idx:])
        
        return split_arrays
    
    def _is_radar_file_valid(self, obsdate: datetime, radar_var: MetRadar) -> bool:
        """
        验证雷达文件是否存在/有效性
        """
        file_directory = os.path.join(
            self.base_path,
            "RADAR",
            radar_var.value
        )

        try:
            if not os.path.exists(file_directory):
                return False
                
            file_names = sorted([file for file in os.listdir(file_directory) if file.endswith('.npy')])

            if len(file_names):
                self.radar_type = file_names[0].split('_')[-2]
            else:
                return False
            
            
            config = get_config()
            date_format = config.get_date_format()
            filename = '_'.join([self.task_id, 'RADA', self.station_id, obsdate.strftime(date_format), self.radar_type, radar_var.value]) + ".npy"
            
            file_path = os.path.join(
                file_directory,
                filename
            )
            
            return os.path.exists(file_path)
        except Exception as e:
            MLOGE(f"验证雷达文件失败: {e}")
            return False

    def _validate_radar_completeness(self, label_file: str) -> bool:
        """
        验证同一时次不同类型雷达文件完整性
        """
        obsdate = self._extract_timestamp_from_label_file(os.path.basename(label_file))
        if not obsdate:
            return False
        
        radar_vars = list(MetRadar)
        for radar_var in radar_vars:
            if not self._is_radar_file_valid(obsdate, radar_var):
                return False
        return True
    
    def _is_nwp_file_valid(self, obsdate: datetime, nwp_var: MetNwp) -> bool:
        """
        验证雷达文件是否存在/有效性
        """
        file_directory = os.path.join(
            self.base_path,
            "NWP",
            nwp_var.name,
        )

        if obsdate.minute < 30:
            obsdate = obsdate.replace(minute=0)
        else:
            obsdate = obsdate.replace(minute=0) + timedelta(hours=1)

        try:
            if not os.path.exists(file_directory):
                return False
            
            config = get_config()
            date_format = config.get_date_format()
            nwp_prefix = config.get_nwp_prefix()
            filename = '_'.join([self.task_id, nwp_prefix, self.station_id, obsdate.strftime(date_format), nwp_var.value]) + ".npy"
            
            file_path = os.path.join(
                file_directory,
                filename
            )
            
            return os.path.exists(file_path)
        except Exception as e:
            MLOGE(f"验证数值预报文件失败: {e}")
            return False
        
    def _validate_nwp_completeness(self, label_file: str) -> bool:
        """
        验证同一时次不同类型NWP文件完整性
        """
        obsdate = self._extract_timestamp_from_label_file(os.path.basename(label_file))
        if not obsdate:
            return False
        
        nwp_vars = list(MetNwp)
        for nwp_var in nwp_vars:
            if not self._is_nwp_file_valid(obsdate, nwp_var):
                return False
        return True

    def get_valid_sequences(self, min_length: int = 40, max_interval_minutes: int = 10) -> List[List[str]]:
        """
        获取符合条件的序列：先数据完整性检验，再进行时间一致性检验
        
        Args:
            min_length: 最小序列长度，默认40
            max_interval_minutes: 时间间隔阈值（分钟），默认10
            
        Returns:
            List[List[str]]: 符合条件的文件序列列表
        """
        # 第一步：数据完整性检验
        valid_files = []
        for file in self.label_files:
            if self._validate_radar_completeness(file) and self._validate_nwp_completeness(file):
                valid_files.append(file)
        # 第二步：对有效文件进行时间一致性检验
        valid_sequences = []
        if len(valid_files) >= min_length:
            split_arrays = self._validate_label_time_consistency_for_files(
                valid_files, max_interval_minutes=max_interval_minutes
            )
            
            # 只保留长度足够的序列
            for sequence in split_arrays:
                if len(sequence) >= min_length:
                    valid_sequences.append(sequence)
        
        return valid_sequences

    def to_samples(self, sample_length: int = 40, sample_interval: int = 10) -> List[List[str]]:
        """
        将有效序列拆分为多个样本，每个样本包含指定数量的连续文件
        
        Args:
            valid_sequences: 有效序列列表
            sample_length: 每个样本的长度，默认40
            sample_interval: 样本间隔，默认10
            
        Returns:
            List[List[str]]: 生成的样本列表
        """
        valid_sequences =  self.get_valid_sequences(min_length=sample_length, max_interval_minutes=20)

        samples = []
        
        for sequence in valid_sequences:
            if len(sequence) < sample_length:
                continue
                
            # 从序列中提取样本，间隔为sample_interval
            for start_idx in range(0, len(sequence) - sample_length + 1, sample_interval):
                end_idx = start_idx + sample_length
                sample = sequence[start_idx:end_idx]
                samples.append(sample)
        
        return samples
    
    def to_infer_sample(self, sample_length: int = 20) -> List[List[str]]:
        lable_files = self._load_label_files(return_full_path=True)
        
        if len(lable_files) < sample_length:
            return []
        
        seq = lable_files[-sample_length:]
        
        return [seq]