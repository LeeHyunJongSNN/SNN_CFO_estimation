import numpy as np

dataset = []

fname = " "
for fname in ["/home/leehyunjong/Dataset_2.4GHz/1kHz_10/vector/"
              "(sine+square)_1kHz_10_vector_0dB_20000.txt"]:

    print(fname)
    f = open(fname, "r", encoding='utf-8-sig')
    linedata = []

    for line in f:
        if line[0] == "#":
            continue

        linedata = [x.replace('i', 'j') for x in line.split()]
        if len(linedata) == 0:
            continue

        dataset.append(linedata)

    f.close()

dataset = np.array(dataset)

fname = "(sine+square)_1kHz_10_vector_0dB_20000.txt"

np.savetxt(fname, dataset, fmt='%s', delimiter=' ')