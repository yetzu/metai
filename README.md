nohup bash run.scwds.convlstm.sh train > train_convlstm_scwds.log 2>&1 &
nohup bash run.scwds.simvp.sh train > train_simvp_scwds.log 2>&1 &

nohup bash run.scwds.simvp.sh train_gan > train_gan_simvp_scwds.log 2>&1 &

watch -n 1 nvidia-smi
/home/dataset-assist-0/code/submit/output/CP2025000081.zip


find /home/dataset-assist-1/SevereWeather_AI_2025/CP/TrainSet/00 -maxdepth 1 -mindepth 1 -type d | xargs -I {} -P 32   rsync -aW --ignore-existing {} ./00