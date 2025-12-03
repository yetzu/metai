import numpy as np

data = np.load("/data/zjobs/SevereWeather_AI_2025/CP/TrainSet/AH/CP_AH_201604021717_Z9559/LABEL/RA/CP_Label_RA_Z9559_20160402-2053.npy")

print(data)
print(data.shape)
print(data.min())
print(data.max())