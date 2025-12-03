import numpy as np
import os

folder = "/data/zjobs/SevereWeather_AI_2025/CP/TrainSet/AH/CP_AH_201604021717_Z9559/LABEL/RA/"

files = []
with os.scandir(folder) as it:
    for entry in it:
        if entry.is_file() and entry.name.endswith('.npy'):
            files.append(entry.name)

files.sort()

for idx, file in enumerate(files):
    print(idx, file)