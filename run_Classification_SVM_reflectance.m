clear
close all;

% set calibration mode
flag_Reflectance=1;

directoryDB='D:\AVIL_HSI_Data\2016-02-19_CNU_NIR_plastic\';

%% Radiometric calibration for reflectance
if flag_Reflectance
    %% load dark data
    fnameHD=[directoryDB 'dark3.hdr']; % 
    fnameRaw=[directoryDB 'dark3.img'];
    disp(['Select Dark Region']);
    [cube1, Wavelength, rct5, selected_bands]=readNIRHSI(fnameHD,fnameRaw,1000);;
    
    [row_2, vertical_2, col_2] = size(cube1); %cube1
    CUBE_ref=zeros(row_2*col_2, vertical_2);
    % re-arrange the cube matrix
    for ROW_1 = 1 : row_2
        for COL_1 = 1 : col_2
            CUBE_ref((ROW_1-1)*col_2+COL_1,:) = cube1(ROW_1, :, COL_1);
        end
    end
    %raw profile for dark
    %R_dark=mean(CUBE_ref);
    R_dark=median(CUBE_ref);
    
    %% load white reference
    fnameHD=[directoryDB 'white3.hdr']; % 
    fnameRaw=[directoryDB 'white3.img'];
    
    [cube1, Wavelength, rct5, selected_bands]=readNIRHSI(fnameHD,fnameRaw,1100);
    
    [row_2, vertical_2, col_2] = size(cube1); %cube1
    CUBE_ref=zeros(row_2*col_2, vertical_2);
    % re-arrange the cube matrix
    for ROW_1 = 1 : row_2
        for COL_1 = 1 : col_2
            CUBE_ref((ROW_1-1)*col_2+COL_1,:) = cube1(ROW_1, :, COL_1);
        end
    end
    %raw profile for dark
    %R_white=mean(CUBE_ref);
    R_white=median(CUBE_ref);
    
    figure(10);clf, set(gcf,'color','w');  hold on;
    plot(Wavelength,R_dark,'b-','linewidth',2);
    plot(Wavelength,R_white,'r-','linewidth',2);
    figure(10); xlabel('Wavelength [nm]','fontsize', 13), ylabel('Digital Number','fontsize',13); box on;
    h=legend('Dark spectrum', 'Whilte reference'); set(h,'fontsize',13);
    
    save R_white R_white
    save R_dark R_dark
end

%% Set data
DBtype=2;

switch(DBtype)
    case 1  
        fnameHD=[directoryDB 'plate1.HDR']; % plastic base
        fnameRaw=[directoryDB 'plate1.IMG'];   
    case 2    
        fnameHD=[directoryDB 'plate3.HDR']; %  rubbber base
        fnameRaw=[directoryDB 'plate3.IMG'];   
end

%% Load HSI raw data
[cube, Wavelength, rct5, selected_bands S]=readNIRHSI(fnameHD,fnameRaw,0);
[row, vertical, col] = size(cube);

%% Preprocessing
med_cube = medcube(cube, 9);                                            % median filter (remove salt noise)
%medSG_cube = sgcube(med_cube11, 2, 5);                                      % smoothing (remove random noise
% 아래 둘중 하나 고르기
%norm_cube = rnrsnorm(medSG_cube);                                           % 최소값 제거 최대값 = 1
remvnoise_cube = noisetozero(med_cube, 0); %2250                                  % 평균값이 일정값 이하면 모든 밴드 0으로 초기화
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
% 컬러셋은 내가 지정한 컬러셋 사용하시길, 검정(1), 빨강(2), 초록(3), 파랑(4), 시안(5), 노랑(6) 순서임
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

%% Extract training DB
showHSI(setcube, selected_bands, 100);
N= ClassNum;
Box= Boxsize;
L= learing_num;
[X, cp, rp] = PLSlearn(setcube, N, Box, fignum, colorset, L);

% reflectance
if flag_Reflectance
    for i=1:size(X,1)
        S=X(i,:);
        X(i,:)=(S-R_dark)./(R_white-R_dark);
    end
end 

%% Classification by Multi-class Adaboost
% Adaboost
Y=[ones(100,1);2*ones(100,1);3*ones(100,1);4*ones(100,1);5*ones(100,1);6*ones(100,1)];
Mdl = fitensemble(X,Y,'AdaBoostM2',100,'Tree')

Predict = predict(Mdl,X)
figure(120); bar(Predict);

% Test for the input
[row_2, vertical_2, col_2] = size(setcube);
result=zeros(row_2,col_2);

for ROW_1 = 1 : row_2
    ROW_1
    xtest=[];
    for COL_1 = 1 : col_2
        R_raw = setcube(ROW_1, :, COL_1);
        if flag_Reflectance
            S=R_raw;
            R_raw=(S-R_dark)./(R_white-R_dark);
        end
        xtest(COL_1,:) = R_raw;
    end
    [ypred] = predict(Mdl, xtest);
    result(ROW_1,:)=ypred;
end
classmap = colorset(1:N,:);
figure(150); imagesc(result); set(gcf,'color','w'); title('Classification by Adaboost','fontsize',14); colorbar;colormap(classmap);


%% Classification by Multi-class SVM을 이용한 분류기 학습
XM=X;
%Y=Indx;

nbclass=N;%length(ID_map);
%[n1, n2]=size(DBtot);%(xapp);
%-----------------------------------------------------
%   Learning and Learning Parameters
c = 1000;
lambda = 1e-7;
kerneloption= 2;
kernel='sam';%'euclidean'%'kl';%'jcb';%'intersection';%'gaussian';'htrbf';'sam'
%kernel='intersection';
verbose = 1;
%---------------------One Against All algorithms----------------
[xsup,w,b,nbsv]=svmmulticlassoneagainstall(XM,Y,nbclass,c,lambda,kernel,kerneloption,verbose);

% save
%save DBclass xsup w b nbsv kernel kerneloption

% Test for the input
xtest=X;%(test,:);
TrueID=Y;%(test,:);

[ypred,maxi] = svmmultival(xtest,xsup,w,b,nbsv,kernel,kerneloption);
figure(200), plot(ypred);
pause(1);

[row_2, vertical_2, col_2] = size(setcube);
result=zeros(row_2,col_2);

for ROW_1 = 1 : row_2
    ROW_1
    xtest=[];
    for COL_1 = 1 : col_2
        R_raw = setcube(ROW_1, :, COL_1);
        if flag_Reflectance
            S=R_raw;
            R_raw=(S-R_dark)./(R_white-R_dark);
        end
        xtest(COL_1,:) = R_raw;
    end
    [ypred,maxi] = svmmultival(xtest,xsup,w,b,nbsv,kernel,kerneloption);
    result(ROW_1,:)=ypred;
end
classmap = colorset(1:N,:);
figure(200); imagesc(result); set(gcf,'color','w'); title('Classification by SVM','fontsize',14); colorbar;colormap(classmap);

%% Classification by Partial Least Squares
number_N = L*N*(Box^2); % 총 학습개수
Y2 = kron(eye(ClassNum),ones((L*(Box^2)),1));
% Normalization 
xmean = mean(X);                                                            % 학습 데이터의 평균
xstd = std(X);                                                              % 학습 데이터의 표준편차
ymean = mean(Y2);
                                                            % 가중치 데이터의 평균
ystd = std(Y2);                                                              % 가중치 데이터의 표준편차
X_norm = (X - xmean(ones(number_N,1),:))./xstd(ones(number_N,1),:);                        % 학습 데이터 - 평균 / 표준편차
Y_norm = (Y2 - ymean(ones(number_N,1),:))./ystd(ones(number_N,1),:);                        % 가중치 데이터 - 평균 / 표준편차 

% Tolerance for 90 percent score
tol = (1-0.9) * 25 * 4;                                                     % 에러 10%까지 
% Perform PLS
[T,P,U,Q,B,W] = pls(X_norm, Y_norm, tol);

if flag_Reflectance
    setcube2=zeros(size(setcube));
    for ROW_1 = 1 : row_2
        for COL_1 = 1 : col_2
            R_raw = setcube(ROW_1, :, COL_1);
            if flag_Reflectance
                S=R_raw;
                R_raw=(S-R_dark)./(R_white-R_dark);
                setcube2(ROW_1,:,COL_1)=R_raw;
            end
        end
    end
else
    setcube2=setcube;
end

Result_image = PLSresult(setcube2, X, Y2, P, B, Q);

classmap = colorset(1:N,:);
figure(300); imagesc(Result_image); colorbar; colormap(classmap);             % 분류 결과 (색깔로 표현)
title('Classification by PLS-DA','fontsize',14);
