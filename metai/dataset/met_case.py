import os
from dataclasses import dataclass, field
from typing import Optional, List
from datetime import datetime, timedelta
from functools import cached_property

from metai.utils import MetLabel, MetRadar, MetNwp
from metai.utils.met_config import get_config, MetConfig

@dataclass
class MetCase:
    """
    天气个例数据结构，基于竞赛数据目录结构设计。
    用于管理单个天气个例的元数据、文件路径和完整性校验。
    """
    case_id: str        # 个例唯一标识符 (如: CP_00_08220804_00093)
    
    # 业务属性
    task_id: str        # 任务类别: CP(短时强降水), TSW(雷暴大风), HA(冰雹)
    region_id: str      # 区域编码 (如: AH, CP)
    station_id: str     # 站点代码 (如: Z9796, 00093)
    base_path: str      # 天气个例数据集根目录路径
    
    radar_type: str = 'SA'     # 雷达类型: SA, SB, SC (通常会自动检测)
    
    # 数据集类型配置
    is_train: bool = True      # 是否为训练集
    test_set: str = "TestSet"  # 测试集类型: "TestA" 或 "TestB"

    # 内部缓存：用于存储文件名到时间戳的映射，避免重复解析
    _timestamp_cache: dict = field(default_factory=dict, repr=False)

    @classmethod
    def create(cls, case_id: str, config: Optional['MetConfig'] = None, **kwargs) -> 'MetCase':
        """
        工厂方法：根据 case_id 创建 MetCase 对象。
        
        Args:
            case_id: 个例唯一标识符
            config: 配置对象，如果为 None 则自动获取
            **kwargs: 其他传递给构造函数的参数
        """
        parts = case_id.split('_')
        if len(parts) != 4:
            raise ValueError(f"[ERROR] Invalid case_id format: {case_id}")
        
        task_id, region_id, _, station_id = parts

        # 获取配置
        if config is None:
            config = get_config()
        
        # 构建基础路径
        base_path = os.path.join(
            config.root_path,
            task_id,
            "TrainSet" if kwargs.get('is_train', True) else kwargs.get('test_set', "TestSet"),
            region_id,
            case_id,
        )
        
        return cls(
            case_id=case_id,
            task_id=task_id,
            region_id=region_id,
            station_id=station_id,
            base_path=base_path,
            **kwargs
        )
    
    @cached_property
    def label_files(self) -> List[str]:
        """
        懒加载属性：获取当前个例所有标签文件名列表。
        仅在首次访问时扫描磁盘，后续访问直接使用缓存。
        """
        return self._load_files("LABEL", MetLabel.RA.name)

    def _load_files(self, category: str, sub_category: str, return_full_path=False) -> List[str]:
        """
        通用的文件加载工具方法。
        使用 os.scandir 替代 os.listdir 以提升性能。
        """
        data_dir = os.path.join(self.base_path, category, sub_category)
        try:
            if not os.path.exists(data_dir):
                return []
            
            files = []
            with os.scandir(data_dir) as it:
                for entry in it:
                    if entry.is_file() and entry.name.endswith('.npy'):
                        files.append(entry.name)
            
            files.sort()
            
            if return_full_path:
                return [os.path.join(data_dir, f) for f in files]
            return files
        except Exception as e:
            print(f"[ERROR] Failed to load files in {category}/{sub_category}: {e}")
            return []

    def _extract_timestamp_from_label_file(self, filename: str) -> Optional[datetime]:
        """
        从文件名中提取时间戳，并进行缓存。
        格式示例: CP_Label_RA_Z9559_20180704-1213.npy
        """
        if filename in self._timestamp_cache:
            return self._timestamp_cache[filename]

        name_without_ext = filename.replace('.npy', '')
        parts = name_without_ext.split('_')
        
        if len(parts) >= 4:
            date_time = parts[-1]
            try:
                config = get_config()
                dt = datetime.strptime(date_time, config.get_date_format())
                self._timestamp_cache[filename] = dt
                return dt
            except ValueError:
                return None
        return None

    def get_valid_sequences(self, min_length: int = 40, max_interval_minutes: int = 10) -> List[List[str]]:
        """
        获取符合条件的序列。
        步骤：
        1. 数据完整性检验 (Radar 和 NWP 文件必须存在)。
        2. 时间一致性检验 (序列中断裂时间不超过 max_interval_minutes)。
        
        Args:
            min_length: 最小序列长度
            max_interval_minutes: 允许的最大时间间隔（分钟）
            
        Returns:
            List[List[str]]: 包含完整路径的文件序列列表
        """
        # 第一步：数据完整性检验
        valid_files = []
        
        # 使用 label_files 属性 (已排序)
        label_dir = os.path.join(self.base_path, "LABEL", MetLabel.RA.name)
        
        for filename in self.label_files:
            file_path = os.path.join(label_dir, filename)
            
            # 检查对应时刻的 Radar 和 NWP 是否齐全
            if self._validate_radar_completeness(file_path) and self._validate_nwp_completeness(file_path):
                valid_files.append(file_path)

        # 第二步：对有效文件进行时间一致性检验
        if len(valid_files) < min_length:
            return []

        return self._split_by_time_consistency(valid_files, min_length, max_interval_minutes)

    def _split_by_time_consistency(self, file_paths: List[str], min_length: int, max_interval_minutes: int) -> List[List[str]]:
        """
        基于时间间隔拆分文件列表。
        如果相邻文件的间隔超过阈值，则断开序列。
        """
        if not file_paths:
            return []

        # 提取时间戳用于排序和计算间隔
        files_with_ts = []
        for fp in file_paths:
            ts = self._extract_timestamp_from_label_file(os.path.basename(fp))
            if ts:
                files_with_ts.append((fp, ts))
        
        # 确保按时间排序
        files_with_ts.sort(key=lambda x: x[1])
        
        valid_sequences = []
        if not files_with_ts:
            return []

        current_seq = [files_with_ts[0][0]]
        
        for i in range(1, len(files_with_ts)):
            curr_path, curr_ts = files_with_ts[i]
            prev_ts = files_with_ts[i-1][1]
            
            # 计算分钟间隔
            diff = (curr_ts - prev_ts).total_seconds() / 60
            
            if diff <= max_interval_minutes:
                current_seq.append(curr_path)
            else:
                # 间隔过大，断开序列
                if len(current_seq) >= min_length:
                    valid_sequences.append(current_seq)
                current_seq = [curr_path]
        
        # 处理最后一个序列
        if len(current_seq) >= min_length:
            valid_sequences.append(current_seq)
            
        return valid_sequences

    def _validate_radar_completeness(self, label_file: str) -> bool:
        """
        验证同一时次不同类型雷达文件的完整性。
        必须包含 MetRadar 枚举中定义的所有变量。
        """
        obsdate = self._extract_timestamp_from_label_file(os.path.basename(label_file))
        if not obsdate:
            return False
        
        radar_vars = list(MetRadar)
        for radar_var in radar_vars:
            if not self._is_radar_file_valid(obsdate, radar_var):
                return False
        return True

    def _is_radar_file_valid(self, obsdate: datetime, radar_var: MetRadar) -> bool:
        """
        验证特定变量的雷达文件是否存在。
        会自动尝试检测雷达类型 (如 SA/SB/SC)。
        """
        file_directory = os.path.join(
            self.base_path,
            "RADAR",
            radar_var.value
        )

        try:
            if not os.path.exists(file_directory):
                return False
            
            # 逻辑保持一致：如果尚未确定雷达类型或为默认值，尝试探测
            # 原版代码每次都会 listdir 取第一个文件，这里优化为按需探测但逻辑结果一致
            if self.radar_type == 'SA': 
                with os.scandir(file_directory) as it:
                    for entry in it:
                        if entry.name.endswith('.npy'):
                            parts = entry.name.split('_')
                            if len(parts) >= 2:
                                # 假设格式 ..._TYPE_VAR.npy，取倒数第二个
                                detected_type = parts[-2]
                                if detected_type in ['SA', 'SB', 'SC', 'CB']:
                                    self.radar_type = detected_type
                            break

            config = get_config()
            date_format = config.get_date_format()
            
            # 构建预期文件名
            # 格式: Task_RADA_Station_Time_Type_Var.npy
            filename = '_'.join([
                self.task_id, 
                'RADA', 
                self.station_id, 
                obsdate.strftime(date_format), 
                self.radar_type, 
                radar_var.value
            ]) + ".npy"
            
            file_path = os.path.join(file_directory, filename)
            return os.path.exists(file_path)
            
        except Exception as e:
            print(f"[ERROR] Failed to validate radar file: {e}")
            return False

    def _validate_nwp_completeness(self, label_file: str) -> bool:
        """
        验证同一时次不同类型数值预报(NWP)文件的完整性。
        """
        obsdate = self._extract_timestamp_from_label_file(os.path.basename(label_file))
        if not obsdate:
            return False
        
        nwp_vars = list(MetNwp)
        for nwp_var in nwp_vars:
            if not self._is_nwp_file_valid(obsdate, nwp_var):
                return False
        return True

    def _is_nwp_file_valid(self, obsdate: datetime, nwp_var: MetNwp) -> bool:
        """
        验证特定变量的 NWP 文件是否存在。
        NWP 时间通常需要对齐到最近的小时。
        """
        file_directory = os.path.join(
            self.base_path,
            "NWP",
            nwp_var.name,
        )

        # NWP 时间对齐逻辑 (与原版保持完全一致)：
        # < 30分 -> 当前小时 :00
        # >= 30分 -> 下一小时 :00
        if obsdate.minute < 30:
            obsdate_aligned = obsdate.replace(minute=0)
        else:
            obsdate_aligned = obsdate.replace(minute=0) + timedelta(hours=1)

        try:
            if not os.path.exists(file_directory):
                return False
            
            config = get_config()
            date_format = config.get_date_format()
            nwp_prefix = config.get_nwp_prefix()
            
            # 构建预期文件名
            # 格式: Task_Prefix_Station_Time_Var.npy
            filename = '_'.join([
                self.task_id, 
                nwp_prefix, 
                self.station_id, 
                obsdate_aligned.strftime(date_format), 
                nwp_var.value
            ]) + ".npy"
            
            file_path = os.path.join(file_directory, filename)
            return os.path.exists(file_path)
            
        except Exception as e:
            print(f"[ERROR] Failed to validate NWP file: {e}")
            return False

    def to_samples(self, sample_length: int = 40, sample_interval: int = 10) -> List[List[str]]:
        """
        将有效序列拆分为多个样本，每个样本包含指定数量的连续文件。
        
        Args:
            sample_length: 每个样本的长度
            sample_interval: 滑动窗口步长
            
        Returns:
            List[List[str]]: 生成的样本列表
        """
        # 获取基础有效序列 
        # 注意：保留原版逻辑，此处强制放宽 max_interval 到 20 分钟
        valid_sequences = self.get_valid_sequences(min_length=sample_length, max_interval_minutes=20)

        samples = []
        for sequence in valid_sequences:
            if len(sequence) < sample_length:
                continue
                
            # 滑动窗口提取样本
            for start_idx in range(0, len(sequence) - sample_length + 1, sample_interval):
                end_idx = start_idx + sample_length
                sample = sequence[start_idx:end_idx]
                samples.append(sample)
        
        return samples
    
    def to_infer_sample(self, sample_length: int = 20) -> List[List[str]]:
        """
        获取用于推理的样本（通常取序列的最后一段）。
        
        Args:
            sample_length: 需要的输入长度
            
        Returns:
            List[List[str]]: 包含一个样本的列表，如果长度不足则返回空列表
        """
        # 注意：这里直接取最新的文件
        label_files = self._load_files("LABEL", MetLabel.RA.name, return_full_path=True)
        
        if len(label_files) < sample_length:
            return []
        
        # 取最后 sample_length 个文件
        seq = label_files[-sample_length:]
        
        return [seq]