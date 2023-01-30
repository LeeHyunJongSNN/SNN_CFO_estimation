import numpy as np
import matplotlib.pyplot as plt

fname = "D:/SNN_dataset/Demodulation/36Mbps/"\
        "802.11_demodulation_36Mbps_21dB_10000_0.txt"

comset = []

raw = np.loadtxt(fname, dtype='float32')

for line in raw:
    line_data = line[0:len(line) - 1]
    bin0 = '0b'
    bit = str(int(line[-1]))
    line_label = int(bin0 + bit, 2)
    comset.append([line_data[0], line_data[1]])

comset = np.array(comset).T
plt.scatter(comset[0], comset[1], c='r', marker='o')
plt.show()
