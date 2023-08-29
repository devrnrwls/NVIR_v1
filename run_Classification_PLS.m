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
% 아래 둘중 하나 고르기
%norm_cube = rnrsnorm(medSG_cube);                                           % 최소값 제거 최대값 = 1
remvnoise_cube = noisetozero(med_cube, 2250);                                   % 평균값이 일정값 이하면 모든 밴드 0으로 초기화
%rnrs_cube = rnrsnorm(remvnoise_cube);

%% base set !!당신은 이것만 설정하면 된다!
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 변수설명;
% 어떤 전처리 결과를 선택하겠는가 *전처리 선택
setcube = remvnoise_cube;                                                   % ★★★ 설정요
%-------------------------------------------------------------------------
% ClassNum : 학습할 개수 ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
ClassNum = 6;                                                               % ★★★ 설정요
%-------------------------------------------------------------------------
% crop_image= 1: 영상 자르기 사용, else: 영상 안자름 ★★★★★★★★★★★★★
% crop_num: 몇개 자를 건지
crop_image = 0;                                                             % ★★★ 설정요
crop_num = 2;                                                               % ★★★ 설정요
%-------------------------------------------------------------------------
% learing_num : 같은 샘플 몇개 학습 할건지
learing_num = 1;                                                            % ★★★ 설정요
%-------------------------------------------------------------------------
% Boxsize : 학습할 위치의 표시 박스의 크기
Boxsize = 10;
%-------------------------------------------------------------------------
% fignum : 학습 한 위치 포지션 나타낼 영상
fignum = 100;
%-------------------------------------------------------------------------
% 컬러셋은 내가 지정한 컬러셋 사용하시길, 검정(1), 빨강(2), 초록(3), 파랑(4), 노랑(5), 시안(6) 순서임
colorset = mycolor;
%-------------------------------------------------------------------------
%% Resize or cropped image !! 샘플을 따로따로 잘라서 한 영상으로 만들고 싶다면.
% (행 열 크기를 동일하기 위해 한 방향으로만 가능하다)
%-------------------------------------------------------------------------
if crop_image == 1                                                                                                         % ★★★ 설정요
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
number_N = L*N*(Box^2); % 총 학습개수
Y = kron(eye(ClassNum),ones((L*(Box^2)),1));
% Normalization 
xmean = mean(X);                                                            % 학습 데이터의 평균
xstd = std(X);                                                              % 학습 데이터의 표준편차
ymean = mean(Y);
                                                            % 가중치 데이터의 평균
ystd = std(Y);                                                              % 가중치 데이터의 표준편차
X_norm = (X - xmean(ones(number_N,1),:))./xstd(ones(number_N,1),:);                        % 학습 데이터 - 평균 / 표준편차
Y_norm = (Y - ymean(ones(number_N,1),:))./ystd(ones(number_N,1),:);                        % 가중치 데이터 - 평균 / 표준편차 

% Tolerance for 90 percent score
tol = (1-0.9) * 25 * 4;                                                     % 에러 10%까지 
% Perform PLS
[T,P,U,Q,B,W] = pls(X_norm, Y_norm, tol);
Result_image = PLSresult(setcube, X, Y, P, B, Q);

classmap = colorset(1:N,:);
figure(300); imagesc(Result_image); colorbar; colormap(classmap);             % 분류 결과 (색깔로 표현)
