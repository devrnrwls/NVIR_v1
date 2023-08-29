%% check test dgist VNIR hyperspectral camera

%% read file
% 
% directoryDB1='D:\(6)_DB_HSI\2017-03-14_VNIR test\'; 
% 
% fnameHD = [directoryDB1 '0314-1612.hdr' ];
% fnameRaw = [directoryDB1 '0314-1612.raw' ];

%% Load HSI raw data
[cube, Wavelength, rct5, selected_bands S]=readDGISTVNIRHSI(0);
[row, vertical, col] = size(cube);

%% check profile

for i = 1:100
    
    figure(3); imagesc(uint8(S)); axis image; truesize;
    disp('Select a pixel');
    [c, r] = getpts(gcf);
    cdata = rand(1,3);
    
    spec = cube(round(r),:,round(c));
    
    
    figure(110); plot(Wavelength, spec, 'color', cdata, 'linewidth', 1); 
    hold on;
    figure(111); imagesc(uint8(S)); hold on; axis image; truesize; plot(c, r, '+', 'color', cdata);
 hold on;
end

% %%
% p_spec = cube(204, :, 39);
% figure(1110); plot(Wavelength, p_spec, 'r-', 'linewidth', 1); 
% 
% f_p_spec = medfilt1(p_spec, 29);
% figure(1111); plot(Wavelength, f_p_spec, 'r-', 'linewidth', 1); 