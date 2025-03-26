colors = ...
 1/256*[ 31,119,180; % 1 默认蓝色
        255,127,14;  % 2 橘色
         44,160,44;  % 3 绿色
        214,39,40;   % 4 红色
        148,103,189; % 5 紫色
        140,86,75;   % 6 棕色 
       227,119,194;  % 7 粉色
       127,127,127;  % 8 灰色
       188,189,34;   % 9 青棕
       23,190,207;   % 10 淡蓝
       26,85,255;    % 11 鲜蓝色
       ];
plot(1:100,rate_robust_ep,'Color', [colors(3,:)],'LineWidth', 1.5)
hold on
plot(1:100,rate_pro_ep,'Color', [colors(11,:)],'LineWidth', 1.5)

% rate_BS_16 : BS
plot(1:100,rate_BS_16 * 0.8,'Color', [colors(7,:)],'LineWidth', 1.5)

% rate_BS_25 : 3D
plot(1:100,rate_BS_25,'Color', [colors(10,:)],'LineWidth', 1.5)

legend( ...
    'TDDRL', ...
    'Static ${\bf \textit Q}$',...
    'BS',...
    '3D mobility',...
    'interpreter','latex',...
    'Location','southeast', ...
    'NumColumns',2,...
    'FontSize',14);
ylim([0 7]);
xlabel('Episodes','FontSize',14);
ylabel('Average sum secrecy rate (bits/s/Hz)','FontSize',14);
grid on
