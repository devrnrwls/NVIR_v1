%% Band selection method
clear
clc;
close all

%% Set data
DBtype=1;
directoryDB1='D:\AVIL_HSI_Data\2016-02-19_CNU_NIR_plastic\';                % Chungnam university
directoryDB2='D:\AVIL_HSI_Data\2016-06-01_Digist_NIR_plastic\';             % DIGIST

switch(DBtype)
    case 1  
        fnameHD=[directoryDB1 '\plate3.HDR'];
        fnameRaw=[directoryDB1 '\plate3.IMG'];   
    case 2    
        fnameHD=[directoryDB2 '\plastic5.HDR'];
        fnameRaw=[directoryDB2 '\plastic5.IMG'];   
end

%% Load HSI raw data
[cube, Wavelength, rct5, selected_bands S]=readNIRHSI(fnameHD,fnameRaw,0);
[row, vertical, col] = size(cube);

%% Preprocessing
med_cube = medcube(cube, 13);   
%cali_cube = calibcube(med_cube, R_dark, R_white);                                                                  
%remvnoise_cube = noisetozero(med_cube, 2250); %2250                                

%% Basic setting
bandselection = 0;
setcube = med_cube;                                                         % Preprocessing selection
ClassNum = 6;                                                               % The number of classes
SameSampleNum = 2;                                                          % The number of same learning samples
BoxSize = 5;                                                                % The size of box displays the loaction to learn sample
LearnFigNum = 100;                                                          % The figure number represent the learned position
ColorSet =mycolor;                                                          % Black(1), Red(2), Green(3), Blue(4), Cyan(5), Yellow(6)
classmap = ColorSet(1:ClassNum,:);

crop_image = 0;                                                             % Whether Image extraction (yes=1)
crop_num = 2;                                                               % The number of image extractions

Y=[ones(SameSampleNum*BoxSize^2,1);2*ones(SameSampleNum*BoxSize^2,1);
    3*ones(SameSampleNum*BoxSize^2,1);4*ones(SameSampleNum*BoxSize^2,1);
    5*ones(SameSampleNum*BoxSize^2,1);6*ones(SameSampleNum*BoxSize^2,1)];

%% Resize or cropped image
if crop_image == 1                                                                                                         % ★★★ 설정요
cropcube = cropcol(setcube, crop_num, S);                                               
setcube=cropcube;
end

%% band selection ** patent start **
if bandselection == 1
    %% Extract training DB : 'X'
    showHSI(setcube, selected_bands, LearnFigNum);
    [X0, cp0, rp0] = TrainDB(setcube, ClassNum, BoxSize, LearnFigNum, ColorSet, SameSampleNum);

    
    %% Test cube 만들기
    %for i = 1:6
    %    TESTsample(i, :) = med_cube(rp(i), :, cp(i));
    %end

    %% Calculation each band subtraction to find best band set
    numeachsample=(BoxSize^2)*SameSampleNum;
    
    sample1 = X0(1:numeachsample, :);
    sample2 = X0(numeachsample+1:numeachsample*2, :);
    sample3 = X0(numeachsample*2+1:numeachsample*3, :);
    sample4 = X0(numeachsample*3+1:numeachsample*4, :);
    sample5 = X0(numeachsample*4+1:numeachsample*5, :);
    sample6 = X0(numeachsample*5+1:numeachsample*6, :);
    
    sample = X0(1:numeachsample*ClassNum, :);
    
    ave1 = mean(sample1);
    ave2 = mean(sample2);
    ave3 = mean(sample3);
    ave4 = mean(sample4);
    ave5 = mean(sample5);
    ave6 = mean(sample6);

    ave_set = [ave1; ave2; ave3; ave4; ave5; ave6];
    combos=combntns(1:6, 2);

    for bands = 1:vertical

        for i = 1:size(combos,1)
            indx1= combos(i,1);
            indx2= combos(i,2);
            sub_DN(i) = abs( ave_set(indx1,bands) - ave_set(indx2,bands) );    
        end

        NCI(bands) = min(sub_DN);
        %max_subDN(bands) = max(sub_DN);
    end

    figure(200); plot(Wavelength, NCI, 'b'); hold on;
    ylabel('Nearest Class Interval (NCI)'); xlabel('Wavelength (nm)');set(gcf,'color','w');
    
    %figure(210); plot(Wavelength, max_subDN, 'b');

    %% find candidate bands by local max (findpeak)
    [pksLocal,locsLocal] = findpeaks(NCI,'MINPEAKHEIGHT',15, 'MINPEAKDISTANCE',5);
    figure(200); hold on; plot(Wavelength(locsLocal), pksLocal, 'ro')

    %% Set cube with band selection
    BandSelection_cube = zeros(row, size(locsLocal, 2), col);
    
    for band = 1:size(locsLocal, 2)
        BandSelection_cube(:, band, :) = med_cube(:, locsLocal(band), :);
    end
    setcube0 = BandSelection_cube;
    

%% find candidate bands by threshold
%     maxNCI = max(NCI);
%     idx_cand = find( NCI > (0.7 * maxNCI) )
%     
%     figure(200); hold on;  plot(Wavelength(idx_cand), NCI(idx_cand), 'ro');
%     numband = size(idx_cand)
%     
%     for band = 1:numband(2)
%         BandSelect_cube(:, band, :) = setcube(:, idx_cand(band), :);
%     end
%     
%     setcube = BandSelect_cube;
%     

end

% ** patent END ** 
%% Extract training DB : 'X'
showHSI(cube, selected_bands, LearnFigNum+1);
[X, cp, rp] = TrainDB(setcube, ClassNum, BoxSize, LearnFigNum+1, ColorSet, SameSampleNum);

%% Extract selected band from X
X0 = zeros(size(X, 1), size(locsLocal, 2));

for band0 = 1:size(locsLocal, 2)
   X0(:, band0) = X(:, locsLocal(band0));
end

%% Display spectra bands

    figure(510); title('Background spectral data'); xlabel('Wavelength[nm]'); ylabel('Digital Number'); hold on; set(gcf,'color','w'); 
    figure(520); title('PVC spectral data'); xlabel('Wavelength[nm]'); ylabel('Digital Number'); hold on; set(gcf,'color','w'); 
    figure(530); title('PE spectral data'); xlabel('Wavelength[nm]'); ylabel('Digital Number'); hold on; set(gcf,'color','w'); 
    figure(540); title('PP spectral data'); xlabel('Wavelength[nm]'); ylabel('Digital Number'); hold on; set(gcf,'color','w'); 
    figure(550); title('PET spectral data'); xlabel('Wavelength[nm]'); ylabel('Digital Number'); hold on; set(gcf,'color','w'); 
    figure(560); title('PS spectral data'); xlabel('Wavelength[nm]'); ylabel('Digital Number'); hold on; set(gcf,'color','w'); 

BACK_SPEC = X(1:50, :);
PVC_SPEC = X(51:100, :);
PE_SPEC = X(101:150, :);
PP_SPEC = X(151:200, :);
PET_SPEC = X(201:250, :);
PS_SPEC = X(251:300, :);

for i = 1:50
    rand_color = rand(1,3);
    
    figure(510); plot(Wavelength, BACK_SPEC(i,:), 'Color', rand_color); 
    figure(520); plot(Wavelength, PVC_SPEC(i,:), 'Color', rand_color); 
    figure(530); plot(Wavelength, PE_SPEC(i,:), 'Color', rand_color); 
    figure(540); plot(Wavelength, PP_SPEC(i,:), 'Color', rand_color);
    figure(550); plot(Wavelength, PET_SPEC(i,:), 'Color', rand_color);
    figure(560); plot(Wavelength, PS_SPEC(i,:), 'Color', rand_color);
    
end



%% Classification by Partial Least Square
% trainDB_num = SameSampleNum*ClassNum*(BoxSize^2);                           % the total number of train samples
% Y_PLS = kron(eye(ClassNum),ones((SameSampleNum*(BoxSize^2)),1));            % Labeling each class
% % normalization X and Y
% xmean = mean(X); xstd = std(X); ymean = mean(Y_PLS); ystd = std(Y_PLS);
% X_norm = (X - xmean(ones(trainDB_num,1),:))./xstd(ones(trainDB_num,1),:);                        
% Y_norm = (Y_PLS - ymean(ones(trainDB_num,1),:))./ystd(ones(trainDB_num,1),:);      
% % Tolerance for 90 percent score
% tol = (1-0.9) * 25 * 4;                                                     
% % Perform PLS
% [T,P,U,Q,B,W] = pls(X_norm, Y_norm, tol);
% Result_PLS = PLSresult(setcube, X, Y_PLS, P, B, Q);
% % Display result
% figure(1100); imagesc(Result_PLS); set(gcf,'color','w');
% title('Classification by PLS-DA','fontsize',14); colorbar; colormap(classmap); 

%% Classification by Multi-class Adaboost
% Y_Ada = Y;
% Mdl = fitensemble(X,Y_Ada,'AdaBoostM2',BoxSize^2,'Tree');
% 
% Predict = predict(Mdl,X);
% figure(1210); bar(Predict); title('Predict bar'); xlabel('the number of samples'); ylabel('labeling number');
% 
% Result_Adaboost = Adaboostresult(setcube, Mdl);
% figure(1200); imagesc(Result_Adaboost); set(gcf,'color','w');
% title('Classification by Adaboost','fontsize',14); colorbar;colormap(classmap);

%% Classification by Mult-class SVM in full bands
%   Learning and Learning Parameters
Y_SVM = Y;
c = 1000;
lambda = 1e-7;
kerneloption= 2;
kernel='sam';%'euclidean'%'kl';%'jcb';%'intersection';%'gaussian';'htrbf';'sam'
verbose = 1;
%---------------------One Against All algorithms----------------
[xsup,w,b,nbsv]=svmmulticlassoneagainstall(X,Y_SVM,ClassNum,c,lambda,kernel,kerneloption,verbose);

[ypred,maxi] = svmmultival(X,xsup,w,b,nbsv,kernel,kerneloption);
figure(1310), plot(ypred);

Result_SVM = SVMresult(setcube, xsup, w, b, nbsv, kernel, kerneloption);
figure(1300); imagesc(Result_SVM); set(gcf,'color','w');
title('Classification by SVM','fontsize',14); colorbar;colormap(classmap);

%% Classification by Mult-class SVM in selected bands
%   Learning and Learning Parameters
Y_SVM = Y;
c = 1000;
lambda = 1e-7;
kerneloption= 2;
kernel='sam';%'euclidean'%'kl';%'jcb';%'intersection';%'gaussian';'htrbf';'sam'
verbose = 1;
%---------------------One Against All algorithms----------------
[xsup,w,b,nbsv]=svmmulticlassoneagainstall(X0,Y_SVM,ClassNum,c,lambda,kernel,kerneloption,verbose);

[ypred,maxi] = svmmultival(X0,xsup,w,b,nbsv,kernel,kerneloption);
figure(1320), plot(ypred);

Result_SVM = SVMresult(setcube0, xsup, w, b, nbsv, kernel, kerneloption);
figure(1310); imagesc(Result_SVM); set(gcf,'color','w');
title('Classification by SVM in selected bands','fontsize',14); colorbar;colormap(classmap);




