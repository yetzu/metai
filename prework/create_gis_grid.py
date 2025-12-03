import os
import sys
import argparse
import pandas as pd
import numpy as np
import rasterio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metai.utils import MLOGE, MLOGI, get_config
from metai.dataset import MetCase

china_lon_min, china_lon_max = 73, 135
china_lat_min, china_lat_max = 18, 54

def read_radar_format(stcd, is_debug=False):
    """
    根据站点代码读取雷达格式配置文件
    
    Args:
        stcd: 站点代码，例如 '9599'
        format_dir: 格式文件目录路径
    
    Returns:
        dict: 包含雷达配置信息的字典，如果文件不存在返回None
    """
    config = get_config(is_debug=is_debug)

    file_path = os.path.join(config.root_path, "FORMAT", "RADAR_Format_File", f"RADA_Format_{stcd}.txt")
            
    if not os.path.exists(file_path):
        MLOGE(f"雷达格式文件不存在: {file_path}")
        return None
    
    config = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # 跳过空行和注释行
                if not line or line.startswith('#') or line.startswith('['):
                    continue
                # 解析键值对
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # 尝试转换为数值类型
                    try:
                        if '.' in value:
                            config[key] = float(value)
                        else:
                            config[key] = int(value)
                    except ValueError:
                        config[key] = value
        
        MLOGI(f"成功读取站点 {stcd} 的雷达格式配置")
        return config
    except Exception as e:
        MLOGE(f"读取雷达格式文件失败 {file_path}: {str(e)}")
        return None

def generate_grid(start_lon, start_lat, end_lon, end_lat, nx, ny):
    """
    生成与matplotlib绘图顺序一致的经纬度网格
    
    Args:
        start_lon: 起始经度（左边界）
        start_lat: 起始纬度（上边界）
        end_lon: 结束经度（右边界）
        end_lat: 结束纬度（下边界）
        nx: 经度方向网格数
        ny: 纬度方向网格数
    
    Returns:
        lon_grid: 经度网格矩阵 (ny, nx)，与matplotlib坐标一致
        lat_grid: 纬度网格矩阵 (ny, nx)，与matplotlib坐标一致
        
    说明:
        - 返回的网格shape为(ny, nx)，其中ny为行数（Y轴），nx为列数（X轴）
        - 纬度从上到下递减（start_lat -> end_lat）
        - 经度从左到右递增（start_lon -> end_lon）
        - 可直接用于plt.imshow(data, extent=[start_lon, end_lon, end_lat, start_lat])
    """
    # 生成经度数组（从左到右递增）
    lon_array = np.linspace(start_lon, end_lon, nx)
    
    # 生成纬度数组（从上到下：start_lat到end_lat）
    # 如果start_lat > end_lat（北半球常见情况），则纬度递减
    lat_array = np.linspace(start_lat, end_lat, ny)
    
    # 使用meshgrid生成网格
    # indexing='xy'：第一个维度对应列（经度），第二个维度对应行（纬度）
    # 返回的数组shape为(ny, nx)
    lon_grid, lat_grid = np.meshgrid(lon_array, lat_array)
    
    return lon_grid, lat_grid

def extract_dem_at_point(dataset, elevation_data, lon, lat, nodata_value=None):
    """
    从DEM数据中提取指定经纬度对应的高程值
    
    Args:
        dataset: rasterio打开的dataset对象
        elevation_data: DEM高程数据数组
        lon: 经度
        lat: 纬度
        nodata_value: NoData值，如果提供则用于标记无效值
    
    Returns:
        float: 高程值（米），如果超出边界或为NoData则返回np.nan
    """
    try:
        # 将经纬度坐标转换为栅格行列索引
        # dataset.index() 返回 (row, col)
        row, col = dataset.index(lon, lat)
        
        # 检查索引是否在有效范围内
        if 0 <= row < dataset.height and 0 <= col < dataset.width:
            elev = elevation_data[row, col]
            # 检查是否为NoData，转换为0
            if nodata_value is not None and elev == nodata_value:
                return 0.0
            return float(elev)
        else:
            return np.nan
    except Exception as e:
        return np.nan

def extract_dem_grid(dataset, elevation_data, lon_grid, lat_grid, nodata_value=None):
    """
    从DEM数据中提取与lon_grid和lat_grid对应的高程数据
    
    Args:
        dataset: rasterio打开的dataset对象
        elevation_data: DEM高程数据数组（二维数组）
        lon_grid: 经度网格矩阵 (ny, nx)
        lat_grid: 纬度网格矩阵 (ny, nx)
        nodata_value: NoData值，如果提供则用于标记无效值
    
    Returns:
        dem_grid: 高程网格矩阵 (ny, nx)，与lon_grid/lat_grid形状一致
    """
    ny, nx = lon_grid.shape
    dem_grid = np.full((ny, nx), np.nan, dtype=np.float32)
    
    # 使用dataset的transform进行坐标转换
    # 反向变换 (~transform) 将地理坐标转换为像素坐标
    transform = dataset.transform
    
    # 将经纬度网格展平以便批量处理
    lons_flat = lon_grid.flatten()
    lats_flat = lat_grid.flatten()
    
    # 批量转换：地理坐标(经度, 纬度) -> 像素坐标(行, 列)
    # 使用 rasterio.transform.rowcol 将地理坐标转换为行列索引
    # rowcol(transform, x, y) 返回 (rows, cols) 即 (行索引列表, 列索引列表)
    # 其中 x 是经度，y 是纬度
    rows, cols = rasterio.transform.rowcol(transform, lons_flat, lats_flat)
    
    # 转换为numpy数组
    rows = np.array(rows)
    cols = np.array(cols)
    
    # 创建有效索引掩码（在DEM边界内）
    valid_mask = (rows >= 0) & (rows < dataset.height) & (cols >= 0) & (cols < dataset.width)
    
    # 提取有效位置的高程值
    if np.any(valid_mask):
        valid_rows = rows[valid_mask]
        valid_cols = cols[valid_mask]
        valid_indices = np.where(valid_mask)[0]
        
        # 从elevation_data中提取高程值（注意：elevation_data是二维数组 [row, col]）
        elev_values = elevation_data[valid_rows, valid_cols].astype(np.float32)
        
        # 检查NoData值并转换为0
        if nodata_value is not None:
            nodata_mask = elev_values == nodata_value
            elev_values[nodata_mask] = 0.0
        
        # 将高程值填充到结果数组中
        dem_grid_flat = dem_grid.flatten()
        dem_grid_flat[valid_indices] = elev_values
        dem_grid = dem_grid_flat.reshape(ny, nx)
    
    return dem_grid

def main():
    parser = argparse.ArgumentParser(description='样本生成处理脚本')
    parser.add_argument('-d', '--debug', action='store_true', help='启用调试模式 (默认: False)')
    args = parser.parse_args()
    
    # 确定是否为调试模式
    is_debug = True if args.debug else False
    config = get_config(is_debug=is_debug)
    files = os.listdir(os.path.join(config.root_path, "FORMAT", "RADAR_Format_File"))
            
    dem_file = os.path.join(config.root_path, "SRTM_China", "China.img")
    with rasterio.open(dem_file) as dataset:
        print(f"成功打开文件: {dem_file}")
            
        # 1. 读取并打印元数据
        print("\n--- 数据集元数据 ---")
        print(f"格式: {dataset.driver}")
        print(f"波段数量: {dataset.count}")
        print(f"图像尺寸 (宽x高): {dataset.width} x {dataset.height} 像素")
        print(f"坐标参考系统 (CRS): {dataset.crs}")
        print(f"地理边界框 (Bounds): {dataset.bounds}")
        print(f"地理变换参数 (Transform): \n{dataset.transform}")
        
        # 获取 NoData 值，这代表没有有效数据的像素值
        nodata_value = dataset.nodata
        print(f"NoData 值: {nodata_value}")
        elevation_data = dataset.read(1)

        mask = elevation_data == nodata_value
            
        # 使用掩码来选择有效数据
        valid_data = elevation_data[~mask]
        print(f"有效数据数量: {valid_data.size}")
        print(f"有效数据范围: {np.min(valid_data)}, {np.max(valid_data)}")

        # 测试提取单个点的高程值
        test_lon, test_lat = 119.48, 29.22
        test_elev = extract_dem_at_point(dataset, elevation_data, test_lon, test_lat, nodata_value)
        print(f"\n测试点 ({test_lon}, {test_lat}) 的高程: {test_elev} 米")

        for file in files[:]:
            filename_without_ext = os.path.splitext(file)[0]
            stcd = filename_without_ext.split('_')[-1]
            print(f"\n处理站点: {stcd}")
            info = read_radar_format(stcd, is_debug)
            if info:
                lon_grid, lat_grid = generate_grid(
                    info['StartLon'], info['StartLat'], 
                    info['EndLon'], info['EndLat'], 
                    info['nx'], info['ny']
                )
                print(f"网格尺寸: {lon_grid.shape}")
                print(f"经度范围: [{info['StartLon']:.4f}, {info['EndLon']:.4f}]")
                print(f"纬度范围: [{info['EndLat']:.4f}, {info['StartLat']:.4f}]")
                
                # 从DEM中提取对应网格的高程数据
                print("正在提取DEM高程数据...")
                dem_grid = extract_dem_grid(dataset, elevation_data, lon_grid, lat_grid, nodata_value)
                
                # 统计提取的DEM数据
                valid_dem = dem_grid[~np.isnan(dem_grid)]
                if len(valid_dem) > 0:
                    print(f"成功提取DEM数据点: {len(valid_dem)} / {dem_grid.size}")
                    print(f"DEM高程范围: [{np.nanmin(dem_grid):.2f}, {np.nanmax(dem_grid):.2f}] 米")
                    print(f"DEM平均高程: {np.nanmean(dem_grid):.2f} 米")
                else:
                    print("警告: 未能提取到有效的DEM数据")
                
                os.makedirs(os.path.join("data", "dem" , stcd), exist_ok=True)
                
                np.save(os.path.join("data", "dem", stcd, f"lat.npy"), lat_grid)
                np.save(os.path.join("data", "dem", stcd, f"lon.npy"), lon_grid)
                np.save(os.path.join("data", "dem", stcd, f"dem.npy"), dem_grid)
            

if __name__ == "__main__":
    main()