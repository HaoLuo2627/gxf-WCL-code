%file_name = "./2021-01-05 12_14_40_with_RIS_4_v5/all_steps.mat";
%file_name = "./2021-01-05 14_47_40/all_steps.mat";
file_name = "./2020-12-06 16_10_17_with_no_RIS/all_steps.mat";
result_RIS_0 = load(file_name);

file_name = "./2021-01-05 12_18_02_with_RIS_4_BS/all_steps.mat";
result_RIS_4 = load(file_name);
acu_4 = [];
for i =1:100
    start = (i-1) * 100 + 1;
    stop = start + 99;
    acu_4(end+1) = sum(result_RIS_4.reward(start:stop));
end

%file_name = "./2021-01-05 12_15_09_with_RIS_9_v5/all_steps.mat";
%file_name = "./2021-01-05 14_48_10/all_steps.mat";
file_name = "./2021-01-05 12_18_19_with_RIS_9_BS/all_steps.mat";
result_RIS_9 = load(file_name);
acu_9 = [];
for i =1:100
    start = (i-1) * 100 + 1;
    stop = start + 99;
    acu_9(end+1) = sum(result_RIS_9.reward(start:stop-1));
end
%file_name = "./2021-01-05 12_15_24_with_RIS_16_v5/all_steps.mat";
%file_name = "./2021-01-05 14_48_20/all_steps.mat";
file_name = "./2021-01-05 12_18_36_with_RIS_16_BS/all_steps.mat";
result_RIS_16 = load(file_name);
acu_16 = [];
for i =1:100
    start = (i-1) * 100 + 1;
    stop = start + 99;
    acu_16(end+1) = sum(result_RIS_16.reward(start:stop));
end

%file_name = "./2021-01-05 12_15_46_with_RIS_25_v5/all_steps.mat";
%file_name = "./2021-01-05 14_48_29/all_steps.mat";
file_name = "./2021-01-05 12_19_01_with_RIS_25_BS/all_steps.mat";
result_RIS_25 = load(file_name);
acu_25 = [];
for i =1:100
    start = (i-1) * 100 + 1;
    stop = start + 99;
    acu_25(end+1) = sum(result_RIS_25.reward(start:stop-50));
end