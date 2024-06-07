stream = fopen('/Users/leehyeonjong/GnuradioProjects/WIFI_signals', 'rb');

thresh = 0.0005;
noise_count = 0;
noise_size = 20000;
noise = [];

while noise_count < noise_size
    v = fread(stream, 2, 'float');
    WV = complex(v(1, 1), v(2, 1));
    amplitude = abs(WV);

    if amplitude < thresh
        noise_count = noise_count + 1;
        v = fread(stream, 320 * 2, 'float');
        WV = complex(v(1:2:end), v(2:2:end));
        noise = cat(1, noise, WV);
    end

end
disp("Complete recording!");

for i = 1:noise_size
    for j = 1:160
        records(j, i) = noise((i - 1) * 160 + j, 1);
    end
end
disp("Complete converting!");
records = records.';

file_name = 'Noise_record_wireless.txt';

writematrix(records, file_name, 'Delimiter', '\t');  
disp("Complete saving!");
