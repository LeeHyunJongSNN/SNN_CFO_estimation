clear;
stream = fopen('/Users/leehyeonjong/GnuradioProjects/WiFi_signals', 'rb');
noise_count = 0;
preamble_count = 0;
thresh = 0.001;  % wired: 0.001, wireless: 0.25

% user parameters
offset_value = 47744;
dataset_size = 20;

preambles = [];

while preamble_count < dataset_size
    v = fread(stream, 2, 'float');
    cv = complex(v(1, 1), v(2, 1));
    amplitude = abs(cv);

    if amplitude < thresh
        noise_count = noise_count + 1;

    elseif amplitude > 0 && noise_count > 1000
        noise_count = 0;
        preamble_count = preamble_count + 1;
        v = fread(stream, 320 * 2, 'float');
        cv = complex(v(1:2:end), v(2:2:end));
        preambles = cat(1, preambles, cv);
    end

end

disp("Complete recording!");

for i = 1:dataset_size
    for j = 1:320
        records(j, i) = preambles((i - 1) * 320 + j, 1);
    end
end

disp("Complete converting!");
record_size = size(records);
offsets = ones(record_size(2), 1) * offset_value;
records = records.';
records = records(:, 1:160);
records = [records offsets];

file_name = append('WiFi_10MHz_Preambles_wired_cfo', '.txt');

% writematrix(records, file_name, 'Delimiter', '\t');  
writematrix(records, file_name, 'WriteMode', 'append', 'Delimiter', '\t');  
disp("Complete saving!");

