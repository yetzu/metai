
from typing import List, Union
from metai.utils import MetConfig, MetLabel, MetRadar, MetNwp, MetGis, MetVar

# 默认通道列表常量 - 保持不变
_DEFAULT_CHANNELS: List[Union[MetLabel, MetRadar, MetNwp, MetGis]] = [
    MetLabel.RA, MetRadar.CR, MetRadar.CAP20, MetRadar.CAP30, MetRadar.CAP40, MetRadar.CAP50, MetRadar.CAP60, MetRadar.CAP70, MetRadar.ET, MetRadar.HBR, MetRadar.VIL,
    MetNwp.WS925, MetNwp.WS700, MetNwp.WS500, MetNwp.Q1000, MetNwp.Q850, MetNwp.Q700, MetNwp.PWAT, MetNwp.PE, MetNwp.TdSfc850, MetNwp.LCL, MetNwp.KI, MetNwp.CAPE,
    MetGis.LAT, MetGis.LON, MetGis.DEM, MetGis.MONTH, MetGis.HOUR,
]

def main():
    print(_DEFAULT_CHANNELS)
    print(len(_DEFAULT_CHANNELS))   

if __name__ == "__main__":
    main()