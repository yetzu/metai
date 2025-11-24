from enum import Enum
from typing import Any, Optional, Union, Dict


class MetBaseEnum(str, Enum):
    """
    通用元枚举基类，继承 str 以便天然支持字符串操作与比较，
    并在枚举成员上携带 min/max/missing_value 元数据以便快捷获取数值范围和缺测值。
    """

    # 为类型检查器声明成员属性
    __slots__ = ("_min", "_max", "_missing_value")
    _min: Optional[float]
    _max: Optional[float]
    _missing_value: Optional[float]

    def __new__(cls, value: str, min_value: Optional[float] = None, max_value: Optional[float] = None, missing_value: Optional[float] = None):
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj._min = min_value
        obj._max = max_value
        obj._missing_value = missing_value
        return obj

    @property
    def min(self) -> Optional[float]:
        return self._min

    @property
    def max(self) -> Optional[float]:
        return self._max

    @property
    def missing_value(self) -> Optional[float]:
        return self._missing_value

    @property
    def limits(self) -> Optional[Dict[str, float | None]]:
        if self._min is None or self._max is None:
            return None
        return {"min": self._min, "max": self._max, "missing_value": self._missing_value}

    @property
    def parent(self) -> str:
        """返回父类别名称"""
        return self.__class__.__name__.replace("Met", "").upper()


class MetLabel(MetBaseEnum):
    """
    标签数据类型枚举
    用于标识气象标签数据的类型
    """
    RA = ("RA", 0, 300, -9)
    # RB = "RB"


class MetRadar(MetBaseEnum):
    """
    雷达数据类型枚举
    包含各种雷达产品的数据类型
    """
    CR = ("CR", 0, 800, -32768)
    CAP20 = ("CAP20", 0, 800, -32768)
    CAP30 = ("CAP30", 0, 800, -32768)
    CAP40 = ("CAP40", 0, 800, -32768)
    CAP50 = ("CAP50", 0, 800, -32768)
    CAP60 = ("CAP60", 0, 800, -32768)
    CAP70 = ("CAP70", 0, 800, -32768)
    ET = ("ET", 0, 150, -1280)
    HBR = ("HBR", 0, 800, -32768)
    VIL = ("VIL", 0, 800, -1280)


class MetNwp(MetBaseEnum):
    """
    NWP (数值天气预报) 数据类型枚举
    包含各种数值天气预报产品的气象要素
    """
    DVG925 = ("DVG925", -1, 1, -9999)
    DVG850 = ("DVG850", -1, 1, -9999)
    DVG200 = ("DVG200", -1, 1, -9999)
    WS925 = ("WS925", 0, 30, -9999)
    WS700 = ("WS700", 0, 40, -9999)
    WS500 = ("WS500", 0, 50, -9999)
    Q1000 = ("Q1000", 0, 30, -9999)
    Q850 = ("Q850", 0, 20, -9999)
    Q700 = ("Q700", 0, 20, -9999)
    RH1000 = ("RH1000", 0, 100, -9999)
    RH700 = ("RH700", 0, 100, -9999)
    RH500 = ("RH500", 0, 100, -9999)
    PWAT = ("PWAT",0, 80, -9999)
    PE = ("PE",0, 100, -9999)
    TdSfc850 = ("TdSfc850", 0, 30, -9999)
    TTdMean74 = ("TTdMean74", 0, 30, -9999)
    TTdMax74 = ("TTdMax74", 0, 30, -9999)
    HTw0 = ("HTw0", 0, 3000, -9999)
    LCL = ("LCL", 0, 3000, -9999)
    muLCL = ("muLCL",0, 3000, -9999)
    KI = ("KI", 0, 40, -9999)
    LI500 = ("LI500", 0, 10000, -9999)
    LI300 = ("LI300", 0, 10000, -9999)
    HT0 = ("HT0", 0, 10000, -9999)
    HT10 = ("HT10", 0, 10000, -9999)
    LFC = ("LFC", 0, 10000, -9999)
    CAPE = ("CAPE", 0, 3000, -9999)


class MetGis(MetBaseEnum):
    """
    GIS (地理信息系统) 数据类型枚举
    包含各种地理和地形数据
    """
    LAT = ("lat", 20, 50)
    LON = ("lon", 90, 130)
    DEM = ("dem", 0, 3000)
    MONTH = ("month", 0, 12)
    HOUR = ("hour", 0, 24)

    # 值域设为 -1 到 1，归一化时会映射到 [0, 1]
    MONTH_SIN = ("month_sin", -1, 1)
    MONTH_COS = ("month_cos", -1, 1)
    HOUR_SIN = ("hour_sin", -1, 1)
    HOUR_COS = ("hour_cos", -1, 1)


class MetVar:
    """
    气象变量容器类
    提供统一的访问接口来获取各种气象数据类型枚举
    """

    class _MetVarAttr:
        """枚举属性代理类，提供对枚举类的透明访问"""

        def __init__(self, enum_class: type, attr_name: str) -> None:
            self.enum_class = enum_class
            self.name = attr_name

        def __getattr__(self, name: str) -> Any:
            """代理所有属性访问到枚举类"""
            return getattr(self.enum_class, name)

        def __repr__(self) -> str:
            """友好的字符串表示"""
            return f"<{self.name} enum proxy: {self.enum_class.__name__}>"

        def __hash__(self) -> int:
            # 以枚举类名和代理名作为哈希，稳定可复现
            return hash((self.enum_class.__name__, self.name))

        def __eq__(self, other: Any) -> bool:
            if not isinstance(other, MetVar._MetVarAttr):
                return False
            return (self.enum_class is other.enum_class) and (self.name == other.name)

    # 各种气象数据类型的枚举代理
    LABEL = _MetVarAttr(MetLabel, "LABEL")
    RADAR = _MetVarAttr(MetRadar, "RADAR")
    NWP = _MetVarAttr(MetNwp, "NWP")
    GIS = _MetVarAttr(MetGis, "GIS")

