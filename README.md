# Center Frequency Offset (CFO) estimation with spiking neural networks
Providing our spiking neural network model for CFO estimation based on ann2snn conversion with benchmark neural network models and conventional methods.
You need SpikingJelly library to use our neural networks for CFO estimation. 
To get a dataset extracted by USRP, please e-mail me (hetzer44@naver.com)

# gr-customs
This folder has a custom GRC block for generating OFDM preambels without data packets in USRP.

# Matlab_files
This folder has matlab files to extract preambles from USRP and save them as datasets. In additino, it contains matlab files to apply Rician and AWGN channel. You can compare the performances of the CFO estimation scheme based on neural networks to the conventional method.
