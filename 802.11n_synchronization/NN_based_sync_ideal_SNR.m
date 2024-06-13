% Reset variables
clear;

% Python process setting
pyenv(ExecutionMode="OutOfProcess")

% NHT Configuration and channel
cfgNHT = wlanNonHTConfig;
cfgNHT.ChannelBandwidth = 'CBW10';
cfgNHT.PSDULength = 100;           

% Create and configure the channel
tgnChannel = wlanTGnChannel;
tgnChannel.DelayProfile = 'Model-B';
tgnChannel.NumTransmitAntennas = 1;
tgnChannel.NumReceiveAntennas = 1;
tgnChannel.TransmitReceiveDistance = 10; % Distance in meters for NLOS
tgnChannel.LargeScaleFadingEffect = 'None';
tgnChannel.NormalizeChannelOutputs = false;

% The number of signals
transmitTime = 1000;

% Set SNR
snr = [0, 9, 18];
S = numel(snr);

% Get the baseband sampling rate
fs = wlanSampleRate(cfgNHT);

% Get the OFDM info
ofdmInfo = wlanNonHTOFDMInfo('NonHT-Data',cfgNHT);

% Set the sampling rate of the channel
tgnChannel.SampleRate = fs;

% Indices for accessing each field within the time-domain packet
ind = wlanFieldIndices(cfgNHT);

% Define error rates
bitErrorRate = zeros(S, 1);
frameErrorRate = zeros(S, 1);

% parfor i = 1:S % Use 'parfor' to speed up the simulation
for i = 1:S % Use 'for' to debug the simulation
    % Set random substream index per iteration to ensure that each
    % iteration uses a repeatable set of random numbers

    stream = RandStream('combRecursive','Seed',0);
    stream.Substream = i;
    RandStream.setGlobalStream(stream);

    % Account for noise energy in nulls so the SNR is defined per
    % active subcarrier
    frameSNR = snr(i)-10*log10(ofdmInfo.FFTLength/ofdmInfo.NumTones);

    % Loop to simulate multiple packets
    rateBitErrors = 0;
    frameSuccess = 0;
    n = 1; % Index of signal transmitted

    whole_txPSDU = [];
    whole_rxPSDU = [];

    while n <= transmitTime
        % Generate a packet waveform
        txPSDU = ones(cfgNHT.PSDULength*8,1); % PSDULength in bytes
        tx = wlanWaveformGenerator(txPSDU,cfgNHT);
        
        % Add trailing zeros to allow for channel filter delay
        tx = [tx; zeros(15,cfgNHT.NumTransmitAntennas)]; %#ok<AGROW>
        
        % Pass the waveform through the TGn channel model 
        reset(tgnChannel); % Reset channel for different realization
        rx = tgnChannel(tx);
                
        % Add noise
        rx = awgn(rx,frameSNR);
 
        % Frame detect and determine coarse frame offset
        coarsePktOffset = wlanPacketDetect(rx,cfgNHT.ChannelBandwidth);
        if isempty(coarsePktOffset) % If empty no L-STF detected
            n = n + 1;
            continue; % Frame detection failed. Go to next loop iteration
        end
        
        % Extract L-STF and perform coarse frequency offset correction
        lstf = rx(coarsePktOffset+(ind.LSTF(1):ind.LSTF(2)),:); 
        % coarseFreqOff = wlanCoarseCFOEstimate(lstf, cfgNHT.ChannelBandwidth);
        
        nn_lstf = [];
        for k = 1:length(lstf)
            nn_lstf = [nn_lstf [real(lstf(k)) imag(lstf(k))]];
        end
        
        nn_lstf = py.numpy.array(nn_lstf);
        coarseFreqOff = pyrunfile("cfo_estimate_scnn.py", "result", mat_input=nn_lstf);
        
        rx = frequencyOffset(rx, fs, -coarseFreqOff);
        
        % Extract the non-HT fields and determine fine frame offset
        nonhtfields = rx(coarsePktOffset+(ind.LSTF(1):ind.LSIG(2)),:); 
        finePktOffset = wlanSymbolTimingEstimate(nonhtfields,...
            cfgNHT.ChannelBandwidth);
        
        % Determine final packet offset
        pktOffset = coarsePktOffset + finePktOffset;

        % If frame detected outwith the range of expected delays from the
        % channel modeling; packet error
        if pktOffset > 15
            n = n + 1;
            continue; % Packet error occurred. Go to next loop iteration.
        end

        % Extract L-LTF and perform fine frequency offset correction and
        % perform channel estimation
        lltf = rx(pktOffset+(ind.LLTF(1):ind.LLTF(2)),:); 
        fineFreqOff = wlanFineCFOEstimate(lltf,cfgNHT.ChannelBandwidth);
        rx = frequencyOffset(rx, fs, -fineFreqOff);
        lltfDemod = wlanLLTFDemodulate(lltf, cfgNHT);
        chanEst = wlanLLTFChannelEstimate(lltfDemod, cfgNHT);
        
        % Extract HT Data samples from the waveform
        nhtdata = rx(pktOffset+(ind.NonHTData(1):ind.NonHTData(2)),:);
        
        % Estimate the noise power in HT data field
        lltfNiose = wlanLLTFNoiseEstimate(lltfDemod);
        
        % Recover the transmitted PSDU in HT Data
        rxPSDU = wlanNonHTDataRecover(nhtdata,chanEst,lltfNiose,cfgNHT);
        if ~isempty(rxPSDU)
            frameSuccess = frameSuccess + 1;
            whole_txPSDU = [whole_txPSDU; txPSDU];
            whole_rxPSDU = [whole_rxPSDU; rxPSDU];
        end

        % Determine if any bits are in error, i.e. bit error and frame error
        bitError = biterr(txPSDU, rxPSDU) / length(txPSDU);
        rateBitErrors = rateBitErrors + bitError;
        n = n + 1;

    end
    
    % Calculate packet error rates (BER and FER) at SNR point
    
    bitErrorRate(i) = rateBitErrors / frameSuccess;
    frameErrorRate(i) = sum(any(biterr(whole_txPSDU, whole_rxPSDU))) / frameSuccess;
    disp(['SNR ' num2str(snr(i))...
          ' completed after '  num2str(n - 1) ' transmitted signals,'...
          ' BER: ' num2str(bitErrorRate(i)) ...
          ' FER: ' num2str(frameErrorRate(i))]);
end

figure;
semilogy(snr,bitErrorRate,'-ob');
grid on;
xlabel('SNR [dB]');
ylabel('BER');
title('802.11n 10MHz NonHT, 1 Channel Model B-NLOS, Bit Error Rate');

figure;
semilogy(snr,frameErrorRate,'-ob');
grid on;
xlabel('SNR [dB]');
ylabel('FER');
title('802.11n 10MHz NonHT, 1 Channel Model B-NLOS, Frame Error Rate');
