import pickle
import numpy as np

fname = "C:/Pycharm BindsNET/WiSig/Full/eq/dataset_2021_03_23_node1-1.pkl"
with open(fname, "rb") as f:
    load_data = pickle.load(f)

print(fname)

complex_form = False

signal = np.array(list(load_data.values()), dtype=list)
temp = np.array([])
wave = []

for i in range(len(signal[0])):
    temp = np.append(temp, signal[0, i].flatten())

if complex_form == True:
    for k in range(int(len(temp) / 2)):
        iq_form = complex(temp[2 * k - 1], temp[2 * k])
        wave.append(iq_form)

    wave_dataset = [wave[i:i + 256] for i in range(0, len(wave), 256)]

    f = open("dataset_2021_03_23_complex.txt", 'w')
    for data in wave_dataset:
        result = " ".join(str(s) for s in data)
        f.write(result)
    f.close()

    print("complete")

if complex_form == False:
    for k in range(int(len(temp) / 2)):
        iq_form = [temp[2 * k - 1], temp[2 * k]]
        wave.append(iq_form)

    wave_dataset = [wave[i:i + 256] for i in range(0, len(wave), 256)]

    f = open("dataset_2021_03_23_divided.txt", 'w')
    for data in wave_dataset:
        result = " ".join(str(s) for s in sum(data, []))
        f.write(result)
    f.close()

    print("complete")