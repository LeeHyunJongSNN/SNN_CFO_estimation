% Reset variables
clear;

% Python process setting
pyenv(ExecutionMode="OutOfProcess")

% NHT Configuration and channel
cfgNHT = wlanNonHTConfig;
cfgNHT.ChannelBandwidth = 'CBW10';
cfgNHT.PSDULength = 100;           

% The number of packets
numSamples = 10000;

% Get the baseband sampling rate
fs = wlanSampleRate(cfgNHT);

% Get the OFDM info
ofdmInfo = wlanNonHTOFDMInfo('NonHT-Data',cfgNHT);

% Tx signals
txPSDU = ones(cfgNHT.PSDULength*8,1);

% Indices for accessing each field within the time-domain packet
ind = wlanFieldIndices(cfgNHT);

% Loop to simulate multiple packets
rateBitErrors = 0;
frameSuccess = 0;

n = 1;

rx_size = 4000;
rx_samples = readmatrix("WiFi_10MHz_record_wireless.txt");
rx_samples = reshape(rx_samples, rx_size, []);
misalignedSize = 400;

whole_txPSDU = [];
whole_rxPSDU = [];

while n <= numSamples

    % Read recorded file from USRPs
    rx = rx_samples(:, n);
 
    % Packet detect and determine coarse packet offset
    coarsePktOffset = wlanPacketDetect(rx,cfgNHT.ChannelBandwidth);
    if isempty(coarsePktOffset) % If empty no L-STF detected; frame error
        n = n + 1;
        continue; % Frame detection failed. Go to next loop iteration
    end
        
    if coarsePktOffset > rx_size - misalignedSize % Severely missaligned frame
        n = n + 1;
        continue; % Frame detection failed. Go to next loop iteration
    end

    % Extract L-STF and perform coarse frequency offset correction
    lstf = rx(coarsePktOffset + (ind.LSTF(1):ind.LSTF(2)), :); 
    % coarseFreqOff = wlanCoarseCFOEstimate(lstf,cfgNHT.ChannelBandwidth);
        
    nn_lstf = [];
    for k = 1:length(lstf)
        nn_lstf = [nn_lstf [real(lstf(k)) imag(lstf(k))]];
    end
    
    nn_lstf = py.numpy.array(nn_lstf);
    coarseFreqOff = pyrunfile("cfo_estimate_scnn.py", "result", mat_input=nn_lstf);
    
    rx = frequencyOffset(rx, fs, -coarseFreqOff);
        
    % Extract the non-HT fields and determine fine packet offset    
    nonhtfields = rx(coarsePktOffset + (ind.LSTF(1):ind.LSIG(2)),:); 
    finePktOffset = wlanSymbolTimingEstimate(nonhtfields,...
        cfgNHT.ChannelBandwidth);
        
    % Determine final packet offset
    pktOffset = coarsePktOffset + finePktOffset;

    % If packet detected outwith the range of expected delays from the
    % channel modeling; packet error
    if pktOffset > 15
        n = n + 1;
        continue; % Packet error occurred. Go to next loop iteration
    end

    % Extract L-LTF and perform fine frequency offset correction and
    % perform channel estimation
    lltf = rx(pktOffset+(ind.LLTF(1):ind.LLTF(2)),:); 
    fineFreqOff = wlanFineCFOEstimate(lltf, cfgNHT.ChannelBandwidth);
    rx = frequencyOffset(rx,fs,-fineFreqOff);
    lltfDemod = wlanLLTFDemodulate(lltf, cfgNHT);
    chanEst = wlanLLTFChannelEstimate(lltfDemod, cfgNHT);
        
    % Extract HT Data samples from the waveform
    nhtdata = rx(pktOffset+(ind.NonHTData(1):ind.NonHTData(2)),:);
        
    % Estimate the noise power in HT data field
    lltfNiose = wlanLLTFNoiseEstimate(lltfDemod);
        
    % Recover the transmitted PSDU in HT Data
    rxPSDU = wlanNonHTDataRecover(nhtdata, chanEst, lltfNiose,cfgNHT);
    if ~isempty(rxPSDU)
        frameSuccess = frameSuccess + 1;
        whole_txPSDU = [whole_txPSDU; txPSDU];
        whole_rxPSDU = [whole_rxPSDU; rxPSDU];
    end

    % Determine if any bits are in error, i.e. bit error, frame error, 
    % and packet error
    bitError = biterr(txPSDU, rxPSDU) / length(txPSDU);
    rateBitErrors = rateBitErrors + bitError;
    n = n + 1;

end
    
% Calculate packet error rates (BER, FER, and PER) at SNR point
bitErrorRate = rateBitErrors / frameSuccess;
frameErrorRate = sum(any(biterr(whole_txPSDU, whole_rxPSDU))) / frameSuccess;
disp([' completed after '  num2str(n-1) ' transmitted signals,'...
      ' BER: ' num2str(bitErrorRate)...
      ' FER: ' num2str(frameErrorRate)]);
