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
% �Ʒ� ���� �ϳ� ����
%norm_cube = rnrsnorm(medSG_cube);                                           % �ּҰ� ���� �ִ밪 = 1
remvnoise_cube = noisetozero(med_cube, 0); %2250                                  % ��հ��� ������ ���ϸ� ��� ��� 0���� �ʱ�ȭ
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
% �÷����� ���� ������ �÷��� ����Ͻñ�, ����(1), ����(2), �ʷ�(3), �Ķ�(4), �þ�(5), ���(6) ������
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


%% Classification by Multi-class SVM�� �̿��� �з��� �н�
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
number_N = L*N*(Box^2); % �� �н�����
Y2 = kron(eye(ClassNum),ones((L*(Box^2)),1));
% Normalization 
xmean = mean(X);                                                            % �н� �������� ���
xstd = std(X);                                                              % �н� �������� ǥ������
ymean = mean(Y2);
                                                            % ����ġ �������� ���
ystd = std(Y2);                                                              % ����ġ �������� ǥ������
X_norm = (X - xmean(ones(number_N,1),:))./xstd(ones(number_N,1),:);                        % �н� ������ - ��� / ǥ������
Y_norm = (Y2 - ymean(ones(number_N,1),:))./ystd(ones(number_N,1),:);                        % ����ġ ������ - ��� / ǥ������ 

% Tolerance for 90 percent score
tol = (1-0.9) * 25 * 4;                                                     % ���� 10%���� 
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
figure(300); imagesc(Result_image); colorbar; colormap(classmap);             % �з� ��� (����� ǥ��)
title('Classification by PLS-DA','fontsize',14);
