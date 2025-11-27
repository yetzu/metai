
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

conda install gxx_linux-64 cuda-toolkit=12.1 -y
conda install pandas matplotlib scipy opencv PyYAML seaborn pydantic cartopy rasterio -y
pip install torch torchvision torchaudio
pip install lightning lightning-utilities tensorboard timm pytorch-msssim

mkdir mamba_build_temp
cd mamba_build_temp
tar -xzf /home/dataset-assist-0/code/mamba_ssm-2.2.6.post3.tar.gz
cd mamba_ssm-2.2.6.post3/
export FORCE_BUILD=1
python setup.py install

pip install mamba-ssm

```

nohup bash run.scwds.simvp.sh train > train_simvp_scwds.log 2>&1 &

nohup bash run.scwds.simvp.sh train_gan > train_gan_simvp_scwds.log 2>&1 &

find /home/dataset-assist-1/SevereWeather_AI_2025/CP/TrainSet/00 -maxdepth 1 -mindepth 1 -type d | xargs -I {} -P 32   rsync -aW --ignore-existing {} ./00

tensorboard --logdir ./output/simvp
watch -n 1 nvidia-smi
/home/dataset-assist-0/code/submit/output/CP2025000081.zip

tensorboard --logdir ./output/meteo_mamba_a800 --port 6006

nohup bash run.scwds.mamba.sh train > train_log.txt 2>&1 &