clear;
stream = fopen('/Users/leehyeonjong/GnuradioProjects/WiFi_signals', 'rb');

packet_count = 0;
dataset_size = 10000;
packet_length = 6000;
signals = [];

while packet_count < dataset_size
    v = fread(stream, packet_length * 2, 'float');
    WV = complex(v(1:2:end), v(2:2:end));
    signals = cat(1, signals, WV);
    packet_count = packet_count + 1;
end
disp("Complete recording!");

for i = 1:dataset_size
    for j = 1:packet_length
        records(j, i) = signals((i - 1) * packet_length + j, 1);
    end
end

disp("Complete converting!");

file_name = 'WiFi_10MHz_record_wireless.txt';

writematrix(records, file_name, 'Delimiter', '\t');  
disp("Complete saving!");
