clear;
raw = readmatrix('wireless/WiFi_10MHz_Preambles_wireless_cfo_rician_18dB.txt');
whole_stf = raw.';
whole_label = raw(:, 161);
estimated = [];

for i = 1:5000
    sample = whole_stf(1:160, i);
    sample = detrend(sample - mean(sample)); % removing dc offset
    nht = wlanNonHTConfig("ChannelBandwidth", "CBW10", ...
        "PSDULength", 100);
    ind = wlanFieldIndices(nht, "L-STF");
    rxLSTF = sample(ind(1):ind(2),:);
    freqOffsetEst = wlanCoarseCFOEstimate(rxLSTF, "CBW10");
    estimated = [estimated; freqOffsetEst];
end

whole_label = sort(whole_label);
estimated = sort(estimated);

disp(round(mae(whole_label, estimated), 2));
