import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
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
    comset.append([line_data[0], line_data[1], line_label])

comset = np.array(comset).T
processed_data = pd.DataFrame({'I': comset[0], 'Q': comset[1], 'Bit': comset[2]})

estimator = KMeans(n_clusters=16)
ids = estimator.fit_predict(processed_data[['I', 'Q', 'Bit']])
plt.tight_layout()
plt.title("K value = {}".format(16))
plt.xlabel('I')
plt.ylabel('Q')
plt.scatter(processed_data['I'], processed_data['Q'], c=ids)
plt.show()
