import torch
import numpy as np
from sklearn.preprocessing import minmax_scale

normalized = []

fname = " "
for fname in ["D:/SNN_dataset/Simple_Waves_RF/"
              "(sine+sawtooth)_1kHz_10_amplitude_18dB_20000.txt"]:

    print(fname)
    f = open(fname, "r", encoding='utf-8-sig')
    linedata = []

    for line in f:
        if line[0] == "#":
            continue

        linedata = [complex(x) for x in line.split()]
        if len(linedata) == 0:
            continue

        linedata_labelremoved = [abs(x) for x in linedata[0:len(linedata) - 1]]
        normalized.append(np.round(linedata_labelremoved, 1))

        converted = torch.tensor(linedata_labelremoved)

    f.close()

normalized = np.array(normalized[0:10000])

np.savetxt("Round", normalized.mean(axis=0), fmt='%s', delimiter=' ')