# prework/step4_create_gisifno.py
import os
import sys
import argparse
import numpy as np
import rasterio
import cv2  # 用于图像/数组缩放

# 将项目根目录添加到系统路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metai.utils import MLOGE, MLOGI, get_config

def read_radar_format(stcd):
    """
    根据站点代码读取雷达格式配置文件
    
    Args:
        stcd: 站点代码，例如 '9599'
    
    Returns:
        dict: 包含雷达配置信息的字典，如果文件不存在返回None
    """
    config_obj = get_config()

    file_path = os.path.join(config_obj.root_path, "FORMAT", "RADAR_Format_File", f"RADA_Format_{stcd}.txt")
            
    if not os.path.exists(file_path):
        MLOGE(f"雷达格式文件不存在: {file_path}")
        return None
    
    radar_config = {}
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
                            radar_config[key] = float(value)
                        else:
                            radar_config[key] = int(value)
                    except ValueError:
                        radar_config[key] = value
        
        MLOGI(f"成功读取站点 {stcd} 的雷达格式配置")
        return radar_config
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
        lon_grid: 经度网格矩阵 (ny, nx)
        lat_grid: 纬度网格矩阵 (ny, nx)
    """
    # 生成经度数组（从左到右递增）
    lon_array = np.linspace(start_lon, end_lon, nx)
    
    # 生成纬度数组（从上到下：start_lat到end_lat）
    lat_array = np.linspace(start_lat, end_lat, ny)
    
    # 使用meshgrid生成网格
    # indexing='xy'：第一个维度对应列（经度），第二个维度对应行（纬度）
    lon_grid, lat_grid = np.meshgrid(lon_array, lat_array)
    
    return lon_grid, lat_grid

def extract_dem_at_point(dataset, elevation_data, lon, lat, nodata_value=None):
    """
    从DEM数据中提取指定经纬度对应的高程值 (单点测试用)
    """
    try:
        # 将经纬度坐标转换为栅格行列索引
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
    从DEM数据中提取与lon_grid和lat_grid对应的高程数据 (批量处理)
    
    Args:
        dataset: rasterio dataset对象
        elevation_data: 内存中的DEM数组
        lon_grid: 经度网格 (ny, nx)
        lat_grid: 纬度网格 (ny, nx)
    
    Returns:
        dem_grid: 高程网格 (ny, nx)
    """
    ny, nx = lon_grid.shape
    dem_grid = np.full((ny, nx), np.nan, dtype=np.float32)
    
    # 获取地理变换参数
    transform = dataset.transform
    
    # 将经纬度网格展平以便批量处理
    lons_flat = lon_grid.flatten()
    lats_flat = lat_grid.flatten()
    
    # 批量转换：地理坐标 -> 像素坐标(行, 列)
    rows, cols = rasterio.transform.rowcol(transform, lons_flat, lats_flat)
    
    rows = np.array(rows)
    cols = np.array(cols)
    
    # 创建有效索引掩码（确保在DEM图像边界内）
    valid_mask = (rows >= 0) & (rows < dataset.height) & (cols >= 0) & (cols < dataset.width)
    
    if np.any(valid_mask):
        valid_rows = rows[valid_mask]
        valid_cols = cols[valid_mask]
        valid_indices = np.where(valid_mask)[0]
        
        # 从elevation_data中提取高程值
        elev_values = elevation_data[valid_rows, valid_cols].astype(np.float32)
        
        # 处理 NoData
        if nodata_value is not None:
            nodata_mask = elev_values == nodata_value
            elev_values[nodata_mask] = 0.0
        
        # 填充回结果数组
        dem_grid_flat = dem_grid.flatten()
        dem_grid_flat[valid_indices] = elev_values
        dem_grid = dem_grid_flat.reshape(ny, nx)
    
    return dem_grid

def resize_grids(lat_grid, lon_grid, dem_grid, target_shape):
    """
    将网格数据缩放到指定大小
    
    Args:
        lat_grid, lon_grid, dem_grid: 原始网格
        target_shape: 目标形状 (height, width)
        
    Note:
        使用最邻近插值 (Nearest Neighbor) 以保持数据的原始离散特性。
    """
    target_h, target_w = target_shape
    # cv2.resize 接受的 dsize 参数顺序为 (width, height)
    dsize = (target_w, target_h)
    
    # 使用最邻近插值 (INTER_NEAREST)
    lat_resized = cv2.resize(lat_grid, dsize, interpolation=cv2.INTER_NEAREST)
    lon_resized = cv2.resize(lon_grid, dsize, interpolation=cv2.INTER_NEAREST)
    dem_resized = cv2.resize(dem_grid, dsize, interpolation=cv2.INTER_NEAREST)
    
    return lat_resized, lon_resized, dem_resized

def main():
    parser = argparse.ArgumentParser(description='GIS信息生成脚本')
    # [新增] 添加 resize 参数，例如 --resize 256 256
    parser.add_argument('--resize', nargs=2, type=int, default=None, metavar=('H', 'W'), 
                        help='将输出网格缩放到指定大小 (例如: --resize 256 256)')
    args = parser.parse_args()
    
    config = get_config()
    format_dir = os.path.join(config.root_path, "FORMAT", "RADAR_Format_File")
    
    if not os.path.exists(format_dir):
        MLOGE(f"未找到雷达格式目录: {format_dir}")
        return

    # 过滤非txt文件
    files = [f for f in os.listdir(format_dir) if f.endswith(".txt")]
            
    dem_file = os.path.join(config.root_path, "SRTM_China", "China.img")
    
    if not os.path.exists(dem_file):
        MLOGE(f"未找到DEM文件: {dem_file}")
        return

    with rasterio.open(dem_file) as dataset:
        print(f"成功打开文件: {dem_file}")
            
        # 1. 打印元数据
        print("\n--- DEM 元数据 ---")
        print(f"格式: {dataset.driver}")
        print(f"波段数: {dataset.count}")
        print(f"尺寸 (宽x高): {dataset.width} x {dataset.height}")
        print(f"坐标系 (CRS): {dataset.crs}")
        print(f"边界 (Bounds): {dataset.bounds}")
        print(f"变换参数 (Transform): \n{dataset.transform}")
        
        nodata_value = dataset.nodata
        print(f"NoData 值: {nodata_value}")
        
        # 加载全量DEM数据到内存
        print("正在将DEM数据加载到内存...")
        try:
            elevation_data = dataset.read(1)
        except MemoryError:
            MLOGE("内存不足: 无法加载完整DEM文件，请尝试分块读取。")
            return

        # 遍历站点
        for file in files:
            filename_without_ext = os.path.splitext(file)[0]
            # 假设文件名格式如 RADA_Format_9599.txt
            parts = filename_without_ext.split('_')
            stcd = parts[-1]
            
            print(f"\n正在处理站点: {stcd}")
            
            # 读取配置
            info = read_radar_format(stcd)
            
            if info:
                # 确保输出目录存在
                output_dir = os.path.join("data", "dem", stcd)
                os.makedirs(output_dir, exist_ok=True)

                # 生成原始网格
                lon_grid, lat_grid = generate_grid(
                    info['StartLon'], info['StartLat'], 
                    info['EndLon'], info['EndLat'], 
                    info['nx'], info['ny']
                )
                
                print(f"原始网格尺寸: {lon_grid.shape}")
                print(f"经度范围: [{info['StartLon']:.4f}, {info['EndLon']:.4f}]")
                print(f"纬度范围: [{info['EndLat']:.4f}, {info['StartLat']:.4f}]")
                
                # 提取DEM
                print("正在提取DEM高程数据...")
                dem_grid = extract_dem_grid(dataset, elevation_data, lon_grid, lat_grid, nodata_value)
                
                # [新增] 检查是否需要缩放
                if args.resize:
                    target_h, target_w = args.resize
                    print(f"正在使用最邻近插值将网格缩放到 {target_h}x{target_w}...")
                    lat_grid, lon_grid, dem_grid = resize_grids(lat_grid, lon_grid, dem_grid, (target_h, target_w))
                
                # 结果统计
                valid_dem = dem_grid[~np.isnan(dem_grid)]
                if len(valid_dem) > 0:
                    print(f"有效数据点: {len(valid_dem)} / {dem_grid.size}")
                    print(f"高程范围: [{np.nanmin(dem_grid):.2f}, {np.nanmax(dem_grid):.2f}] 米")
                    print(f"平均高程: {np.nanmean(dem_grid):.2f} 米")
                else:
                    print("警告: 未提取到有效的DEM数据")
                
                # 保存
                np.save(os.path.join(output_dir, "lat.npy"), lat_grid)
                np.save(os.path.join(output_dir, "lon.npy"), lon_grid)
                np.save(os.path.join(output_dir, "dem.npy"), dem_grid)
                print(f"已保存数据至: {output_dir}")

if __name__ == "__main__":
    main()