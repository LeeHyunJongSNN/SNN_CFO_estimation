import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import detrend
import random

processed_dataset = []
classes = []

fname = " "
for fname in ["/home/leehyunjong/Wi-Fi_Preambles/"
              "20_Shifted_WIFI_10MHz_IQvector_18dB_20000.txt"]:

    print(fname)
    f = open(fname, "r", encoding='utf-8-sig')
    linedata = []

    for line in f:
        if line[0] == "#":
            continue

        linedata = [complex(x) for x in line.split()]
        if len(linedata) == 0:
            continue

        linedata_dcremoved = detrend(linedata - np.mean(linedata))  # removing DC offset
        # linedata_fft = np.fft.fft([x for x in linedata_dcremoved[0:len(linedata_dcremoved) - 1]])

        linedata_fft_1 = np.fft.fft([x for x in linedata_dcremoved[16:80]]) / 64
        linedata_fft_2 = np.fft.fft([x for x in linedata_dcremoved[96:160]]) / 64
        linedata_fft_3 = np.fft.fft([x for x in linedata_dcremoved[192:256]]) / 64
        linedata_fft_4 = np.fft.fft([x for x in linedata_dcremoved[256:len(linedata_dcremoved) - 1]]) / 64
        linedata_fft = np.array(linedata_fft_1.tolist() + linedata_fft_2.tolist() +
                                linedata_fft_3.tolist() + linedata_fft_4.tolist())

        linedata_abs = [abs(x) for x in linedata_fft[0:len(linedata_fft) - 1]]

        processed_dataset.append(linedata_abs)

    f.close()

seed = random.randint(0, int(len(processed_dataset) / 2) - 1)
print(seed)
plt.plot(processed_dataset[seed])
plt.show()
