
# env

```bash
conda config --add channels defaults
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch
conda config --set show_channel_urls yes
conda config --show channels

pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

conda create -n metai python=3.11
conda activate metai

conda install pandas matplotlib scipy opencv-python PyYAML seaborn pydantic cartopy rasterio opencv-python -y
pip install torch torchvision torchaudio --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install lightning lightning-utilities tensorboard timm pytorch-msssim
conda install -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/ gxx_linux-64 -y

conda install -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/ cuda-toolkit=12.1 -y

pip install mamba_ssm


```

nohup bash run.scwds.convlstm.sh train > train_convlstm_scwds.log 2>&1 &
nohup bash run.scwds.simvp.sh train > train_simvp_scwds.log 2>&1 &

nohup bash run.scwds.simvp.sh train_gan > train_gan_simvp_scwds.log 2>&1 &

watch -n 1 nvidia-smi
/home/dataset-assist-0/code/submit/output/CP2025000081.zip


find /home/dataset-assist-1/SevereWeather_AI_2025/CP/TrainSet/00 -maxdepth 1 -mindepth 1 -type d | xargs -I {} -P 32   rsync -aW --ignore-existing {} ./00

tensorboard --logdir ./output/simvp