clear
close all;

%% Set data
DBtype=3;
switch(DBtype)
    case 1  
        fnameHD='D:\AVIL_HSI_Data\2016-02-19 NIR_HSI\plate1.HDR';
        fnameRaw='D:\AVIL_HSI_Data\2016-02-19 NIR_HSI\plate1.IMG';   
    case 2    
        fnameHD='D:\AVIL_HSI_Data\2016-02-19 NIR_HSI\plate3.HDR';
        fnameRaw='D:\AVIL_HSI_Data\2016-02-19 NIR_HSI\plate3.IMG';   
        
    case 3    
        fnameHD='D:\AVIL_HSI_Data\2016-06-01 plastic\plastic5.HDR';
        fnameRaw='D:\AVIL_HSI_Data\2016-06-01 plastic\plastic5.IMG';
        
    case 4    
        fnameHD='D:\AVIL_HSI_Data\2016-06-01 plastic\plastic2.HDR';
        fnameRaw='D:\AVIL_HSI_Data\2016-06-01 plastic\plastic2.IMG'; 
        
    case 5   
        fnameHD='D:\AVIL_HSI_Data\2016-06-01 plastic\plastic3.HDR';
        fnameRaw='D:\AVIL_HSI_Data\2016-06-01 plastic\plastic3.IMG'; 
end

%% Load HSI raw data
[cube, Wavelength, rct5, selected_bands S]=readNIRHSI(fnameHD,fnameRaw,0);
[row, vertical, col] = size(cube);

%% Preprocessing
med_cube = medcube(cube, 13);                                            % median filter (remove salt noise)
%medSG_cube = sgcube(med_cube11, 2, 5);                                      % smoothing (remove random noise
% �Ʒ� ���� �ϳ� ����
%norm_cube = rnrsnorm(medSG_cube);                                           % �ּҰ� ���� �ִ밪 = 1
remvnoise_cube = noisetozero(med_cube, 2250);                                   % ��հ��� ������ ���ϸ� ��� ��� 0���� �ʱ�ȭ
%rnrs_cube = rnrsnorm(remvnoise_cube);

%% base set !!����� �̰͸� �����ϸ� �ȴ�!
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ��������;
% � ��ó�� ����� �����ϰڴ°� *��ó�� ����
setcube = remvnoise_cube;                                                   % �ڡڡ� ������
%-------------------------------------------------------------------------
% ClassNum : �н��� ���� �ڡڡڡڡڡڡڡڡڡڡڡڡڡڡڡڡڡڡڡڡڡڡڡڡڡڡڡڡ�
ClassNum = 6;                                                               % �ڡڡ� ������
%-------------------------------------------------------------------------
% crop_image= 1: ���� �ڸ��� ���, else: ���� ���ڸ� �ڡڡڡڡڡڡڡڡڡڡڡڡ�
% crop_num: � �ڸ� ����
crop_image = 0;                                                             % �ڡڡ� ������
crop_num = 2;                                                               % �ڡڡ� ������
%-------------------------------------------------------------------------
% learing_num : ���� ���� � �н� �Ұ���
learing_num = 1;                                                            % �ڡڡ� ������
%-------------------------------------------------------------------------
% Boxsize : �н��� ��ġ�� ǥ�� �ڽ��� ũ��
Boxsize = 10;
%-------------------------------------------------------------------------
% fignum : �н� �� ��ġ ������ ��Ÿ�� ����
fignum = 100;
%-------------------------------------------------------------------------
% �÷����� ���� ������ �÷��� ����Ͻñ�, ����(1), ����(2), �ʷ�(3), �Ķ�(4), ���(5), �þ�(6) ������
colorset = mycolor;
%-------------------------------------------------------------------------
%% Resize or cropped image !! ������ ���ε��� �߶� �� �������� ����� �ʹٸ�.
% (�� �� ũ�⸦ �����ϱ� ���� �� �������θ� �����ϴ�)
%-------------------------------------------------------------------------
if crop_image == 1                                                                                                         % �ڡڡ� ������
cropcube = cropcol(setcube, crop_num, S);                                               
setcube=cropcube;
end
showHSI(setcube, selected_bands, 100);

%% Learning
showHSI(setcube, selected_bands, 100);
N= ClassNum;
Box= Boxsize;
L= learing_num;

[X, cp, rp] = PLSlearn(setcube, N, Box, fignum, colorset, L);
% Label
Y=[ones(100,1);2*ones(100,1);3*ones(100,1);4*ones(100,1);5*ones(100,1);6*ones(100,1)];
Mdl = fitensemble(X,Y,'AdaBoostM2',100,'Tree')

Predict = predict(Mdl,X)
figure(500); bar(Predict);

%% Partial Least Squares
number_N = L*N*(Box^2); % �� �н�����
Y = kron(eye(ClassNum),ones((L*(Box^2)),1));
% Normalization 
xmean = mean(X);                                                            % �н� �������� ���
xstd = std(X);                                                              % �н� �������� ǥ������
ymean = mean(Y);
                                                            % ����ġ �������� ���
ystd = std(Y);                                                              % ����ġ �������� ǥ������
X_norm = (X - xmean(ones(number_N,1),:))./xstd(ones(number_N,1),:);                        % �н� ������ - ��� / ǥ������
Y_norm = (Y - ymean(ones(number_N,1),:))./ystd(ones(number_N,1),:);                        % ����ġ ������ - ��� / ǥ������ 

% Tolerance for 90 percent score
tol = (1-0.9) * 25 * 4;                                                     % ���� 10%���� 
% Perform PLS
[T,P,U,Q,B,W] = pls(X_norm, Y_norm, tol);
Result_image = PLSresult(setcube, X, Y, P, B, Q);

classmap = colorset(1:N,:);
figure(300); imagesc(Result_image); colorbar; colormap(classmap);             % �з� ��� (����� ǥ��)
