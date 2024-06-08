clear;
raw = readmatrix('WiFi_10MHz_Preambles_wired_cfo_rician.txt');
noise = readmatrix('Noise_10MHz_record_wired.txt');
noise = noise(1:5000, :);
clean = raw(:, 1:160) - noise;    % removing noise samples
noise_added = [];

% user parameter
SNR = 18;

for i = 1:5000
    signal_sample = clean(i, 1:160);
    noise_sample = noise(i, 1:160);

    signal_power = mean(abs(signal_sample).^2);
    noise_power = mean(abs(noise_sample).^2);
    const_factor = sqrt(signal_power / (noise_power * 10^(SNR / 10)));

    signal_sample = signal_sample + const_factor * noise_sample;

    offset = raw(i, 161);
    noise_added = [noise_added; [signal_sample offset]];
end

base_name = 'WiFi_10MHz_Preambles_wired_cfo_rician_';
SNR_name = num2str(SNR);
file_name = append(base_name, SNR_name);
file_name = append(file_name, 'dB.txt');

writematrix(noise_added, file_name, 'Delimiter', '\t');
