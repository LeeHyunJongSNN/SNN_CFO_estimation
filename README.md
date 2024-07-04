# Center Frequency Offset (CFO) estimation with spiking neural networks
Providing our spiking neural network model for CFO estimation based on ann2snn conversion with benchmark neural network models and conventional methods.
You need SpikingJelly library to use our neural networks for CFO estimation. 
To get a dataset extracted by USRP, please e-mail me (hetzer44@naver.com)

# gr-customs
This folder has a custom GRC block for generating OFDM preambels without data packets in USRP.

# train
This folder has train files for neural network models to learn CFO values form STF and LTF datasets.

# test
This folder has test files to calculate MAE of the baseline, benchmarks, and proposed mechanism.

# ofdmSynchronization
This folder has matlab files to calculate BER and FER of the baseline, benchmarks, and proposed mechanism. It has 802.11n Non-HT synchronization flow by MATLAB.
