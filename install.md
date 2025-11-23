# 1. 先退出 metai 环境（避免在子目录创建新环境）
conda deactivate

# 2. 创建临时下载环境，指定独立路径（比如 ~/temp_download_env）
# 同时指定 Python 版本（和原 metai 环境一致，从原环境获取版本）
PYTHON_VERSION=$(conda run -n metai python --version | cut -d' ' -f2 | cut -d'.' -f1-2)
conda create --prefix ~/temp_download_env python=$PYTHON_VERSION -y

conda activate ~/temp_download_env

# 2. 确保离线包目录已创建
mkdir -p ~/code/packages/conda_offline_pkgs

# 3. 进入之前导出清单文件的目录（假设 conda_explicit.txt 在 ~/code/packages/）
cd ~/code/packages/

conda clean -afy

conda install --file conda_explicit.txt --download-only --no-deps -y

# Conda 缓存目录（默认路径，直接复制使用）
CONDA_CACHE_DIR=/home/yyj/opt/anaconda3/pkgs

mkdir -p conda_offline_pkgs
