import pickle
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import detrend

fname = "C:/Pycharm BindsNET/WiSig/Full/dataset_2021_03_23_node23-1.pkl"
with open(fname, "rb") as f:
    load_data = pickle.load(f)

print(fname)

signal = np.array(list(load_data.values()), dtype=list)
temp = np.array([])
wave = []

for i in range(len(signal[0])):
    temp = np.append(temp, signal[0, i].flatten())

for k in range(int(len(temp) / 2)):
    iq_form = complex(temp[2 * k - 1], temp[2 * k])
    wave.append(iq_form)

wave_dataset = [wave[i:i + 256] for i in range(0, len(wave), 256)]
processed_dataset = []

# linedata_fft = np.fft.fft([x for x in linedata[0:len(linedata) - 1]])

for linedata in wave_dataset:
    linedata_dcremoved = detrend(linedata - np.mean(linedata))  # removing DC offset

    linedata_fft_1 = np.fft.fft([x for x in linedata[0:64]]) / 64
    linedata_fft_2 = np.fft.fft([x for x in linedata[64:128]]) / 64
    linedata_fft_3 = np.fft.fft([x for x in linedata[128:192]]) / 64
    linedata_fft_4 = np.fft.fft([x for x in linedata[192:len(linedata) - 1]]) / 64
    linedata_fft = np.array(linedata_fft_1.tolist() + linedata_fft_2.tolist() +
                            linedata_fft_3.tolist() + linedata_fft_4.tolist())

    linedata_abs = [abs(x) for x in linedata_fft[0:len(linedata_fft) - 1]]

    processed_dataset.append(linedata_abs)

plt.plot(processed_dataset[0])
plt.show()