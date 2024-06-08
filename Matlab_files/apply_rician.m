clear;

% Load STF samples
base_name = 'WiFi_10MHz_Preambles_wired_cfo';
file_name = append(base_name, '.txt');
% path = append('wireless/', file_name);
raw = readmatrix(file_name);

% Defining multipath channel
fs = 10e6;
multipathChannel = comm.RicianChannel(...
    'SampleRate', fs, ...
    'PathDelays', [0 1.8 3.4] / fs, ...
    'AveragePathGains',[0 -2 -10], ...
    'KFactor', 4, ...
    'MaximumDopplerShift', 4);

multipath_applied = [];

for i = 1:5000
    sample = raw(i, 1:160).';
    offset = raw(i, 161);
    reset(multipathChannel);
    applied_sample = multipathChannel(sample).';

    multipath_applied = [multipath_applied; [applied_sample offset]];
end

save_name = append(base_name, '_rician');
save_name = append(save_name, '.txt');

writematrix(multipath_applied, save_name, 'Delimiter', '\t');


% nexttile
% plot(abs(sample));
% title('Original');
% 
% nexttile
% plot(abs(outMultipathChan));
% title('Multipath');
