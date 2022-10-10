import numpy as np

shifting = 20

dataset = []

fname = " "
for fname in ["/home/leehyunjong/Wi-Fi_Preambles/"
              "WIFI_10MHz_IQvector_18dB_20000.txt"]:

    print(fname)
    f = open(fname, "r", encoding='utf-8-sig')
    linedata = []

    for line in f:
        if line[0] == "#":
            continue

        linedata = [x for x in line.split()]
        if len(linedata) == 0:
            continue

        dataset.append(linedata)

    f.close()

dataset = np.array(dataset)
whole = np.size(dataset, 0)
half  = int(whole / 2)

dataset[0:half, shifting:320] = dataset[0:half, 0:320 - shifting]
dataset[0:half, 0:shifting] = dataset[half:whole, 0:shifting]

fname = str(shifting) + "_Shifted_WIFI_10MHz_IQvector_18dB_20000.txt"

np.savetxt(fname, dataset, fmt='%s', delimiter=' ')
