import numpy as np
from sklearn.preprocessing import minmax_scale

gamma = 0.1
mae_3 = 1 / np.array([1854.6, 1265.4, 1134.95, 2238.58, 1047.54, 1169.04, 2353.39, 1943.33, 1304.52, 1794.97, 1972.41, 1106.25, 2073.61, 1690.06])
energy = 1 / np.array([2307.35, 266.16, 2123.22, 1125.99, 293.03, 1259.81, 217.41, 2098.89, 26.21, 218.7, 1050.68, 80.56, 10.47, 14.46])

mae_3 = minmax_scale(mae_3)
energy = minmax_scale(energy)

mae_3a = mae_3[4]
energy_a = energy[4]
mae_3b = mae_3[11]
energy_b = energy[11]

eval = mae_3 + gamma * energy
eval_a = mae_3a + gamma * energy_a
eval_b = mae_3b + gamma * energy_b

print(np.argmax(eval))
print(eval_a, eval_b)
