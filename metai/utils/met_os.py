import os

def scan_directory(folder_path, level=1, return_full_path=False):
    """
    递归扫描目录结构
    
    Args:
        folder_path: 要扫描的目录路径
        level: 扫描深度 (0表示只返回路径本身)
        return_full_path: True返回完整路径，False返回文件名
    
    Returns:
        当level<=1时：目录列表
        当level>1时：扁平化的目录列表（所有子目录合并为一个列表）
    """
    if level <= 0:
        return [folder_path] if return_full_path else os.path.basename(folder_path)
    
    # 获取所有子目录并排序
    try:
        folders = sorted([
            os.path.join(folder_path, item) 
            for item in os.listdir(folder_path) 
            if os.path.isdir(os.path.join(folder_path, item))
        ])
    except (OSError, PermissionError):
        return []
    
    if level == 1:
        # level=1时，直接返回当前层级的目录
        if return_full_path:
            return folders
        else:
            return [os.path.basename(folder) for folder in folders]
    else:
        # level>1时，递归获取所有子目录并合并成一个扁平列表
        result = []
        for folder in folders:
            sub_result = scan_directory(folder, level - 1, return_full_path)
            if isinstance(sub_result, list):
                result.extend(sub_result)
            else:
                result.append(sub_result)
        return result