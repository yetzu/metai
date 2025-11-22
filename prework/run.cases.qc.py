import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metai.utils import scan_directory
from metai.dataset import MetCase
from metai.utils import MetVar, MetLabel, MetRadar, MetNwp, get_config

def parse_variable_type(variable_str: str):
    """
    解析变量类型字符串，返回对应的变量对象
    
    Args:
        variable_str: 变量类型字符串 (如 "RA", "CR", "CAP20", "DVG925" 等)
    
    Returns:
        对应的变量对象，如果不支持则返回None
    """
    # MetLabel 变量映射
    metlabel_vars = {
        'RA': MetLabel.RA,
    }
    
    # MetRadar 变量映射
    metradar_vars = {
        'CR': MetRadar.CR,
        'CAP20': MetRadar.CAP20,
        'CAP30': MetRadar.CAP30,
        'CAP40': MetRadar.CAP40,
        'CAP50': MetRadar.CAP50,
        'CAP60': MetRadar.CAP60,
        'CAP70': MetRadar.CAP70,
        'ET': MetRadar.ET,
        'HBR': MetRadar.HBR,
        'VIL': MetRadar.VIL,
    }
    
    # MetNwp 变量映射
    metnwp_vars = {
        'DVG925': MetNwp.DVG925,
        'DVG850': MetNwp.DVG850,
        'DVG200': MetNwp.DVG200,
        'WS925': MetNwp.WS925,
        'WS700': MetNwp.WS700,
        'WS500': MetNwp.WS500,
        'Q1000': MetNwp.Q1000,
        'Q850': MetNwp.Q850,
        'Q700': MetNwp.Q700,
        'RH1000': MetNwp.RH1000,
        'RH700': MetNwp.RH700,
        'RH500': MetNwp.RH500,
        'PWAT': MetNwp.PWAT,
        'PE': MetNwp.PE,
        'TdSfc850': MetNwp.TdSfc850,
        'TTdMean74': MetNwp.TTdMean74,
        'TTdMax74': MetNwp.TTdMax74,
        'HTw0': MetNwp.HTw0,
        'LCL': MetNwp.LCL,
        'muLCL': MetNwp.muLCL,
        'KI': MetNwp.KI,
        'LI500': MetNwp.LI500,
        'LI300': MetNwp.LI300,
        'HT0': MetNwp.HT0,
        'HT10': MetNwp.HT10,
        'CAPE': MetNwp.CAPE,
    }
    
    # 首先检查 MetLabel
    if variable_str.upper() in metlabel_vars:
        return metlabel_vars[variable_str.upper()]
    
    # 然后检查 MetRadar
    if variable_str.upper() in metradar_vars:
        return metradar_vars[variable_str.upper()]
    
    # 最后检查 MetNwp
    if variable_str.upper() in metnwp_vars:
        return metnwp_vars[variable_str.upper()]
    
    return None

def main():
    parser = argparse.ArgumentParser(description='变量极值分析脚本')
    parser.add_argument('-d','--debug', action='store_true', help='启用调试模式 (默认: False)')
    parser.add_argument('-v','--version', type=str, default="v1")  
    parser.add_argument('-x', '--variable', type=str, default="RA", help='变量类型，支持MetLabel(RA)、MetRadar(CR,CAP20-70,ET,HBR,VIL)和MetNwp(各种气象要素) (默认: RA)')
    
     
    args = parser.parse_args()
    
    is_debug = True if args.debug else False
    version = args.version
    # 解析变量类型
    variable = parse_variable_type(args.variable)
    
    # config = get_config(is_debug=is_debug)
    
    file_path = os.path.join("data", version, "statistics", variable.parent, variable.value + ".csv")
    
    df = pd.read_csv(file_path, header=None, names=['case_id', 'min', 'max', "length", 'unnormal_indices'])
    
    df['min'] = pd.to_numeric(df['min'], errors='coerce')
    df['max'] = pd.to_numeric(df['max'], errors='coerce')
    
    if variable.parent == "LABEL":
        condition = (df['max'] <= 0) | (df['max'] > 400) | (df['unnormal_indices'].notna())
    elif variable.parent == "RADAR":
        if variable == MetRadar.CR:
            condition = (df['max'] <= 0) | (df['max'] > 1200) | (df['unnormal_indices'].notna())
        elif variable == MetRadar.ET:
            condition = (df['max'] <= 0) | (df['max'] > 1000)
        elif variable == MetRadar.VIL:
            condition = (df['max'] <= 0) | (df['max'] > 2000)
        else:
            condition = (df['max'] < 0) | (df['max'] > 1200)    
    elif variable.parent == "NWP":
        raise ValueError("NWP variable is not supported")
    
    filtered_df = df[condition]
    
    print(f"\n=== 筛选结果 ===")
    print(f"总记录数: {len(df)}")
    print(f"满足条件的记录数: {len(filtered_df)}")
    print(f"筛选比例: {len(filtered_df)/len(df)*100:.2f}%")
    
    print(f"\n=== 筛选出的数据 ===")
    print(filtered_df)
    
    output_path = os.path.join("data", version, "qc", variable.value + ".csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    filtered_df.to_csv(output_path, index=False)
    print(f"\n筛选结果已保存到: {output_path}")
    
if __name__ == "__main__":
    main()