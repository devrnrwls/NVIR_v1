%% run_classification_plastic_ver2
% 
% readNIRHSI / medcube / noisetozero / cropcol / mycolor / pls / PLSresult
% showHSI
%% Initialized command window
clear
close all;
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
remvnoise_cube = noisetozero(med_cube, 2250); %2250                                

%% Basic setting
setcube = med_cube;                                                             % Preprocessing selection
ClassNum = 6;                                                               % The number of classes
SameSampleNum = 1;                                                          % The number of same learning samples
BoxSize = 10;                                                               % The size of box displays the loaction to learn sample
LearnFigNum = 100;                                                          % The figure number represent the learned position
ColorSet =mycolor;                                                          % Black(1), Red(2), Green(3), Blue(4), Cyan(5), Yellow(6)
classmap = ColorSet(1:ClassNum,:);

crop_image = 0;                                                             % Whether Image extraction (yes=1)
crop_num = 2;                                                               % The number of image extractions

Y=[ones(SameSampleNum*BoxSize^2,1);2*ones(SameSampleNum*BoxSize^2,1);
    3*ones(SameSampleNum*BoxSize^2,1);4*ones(SameSampleNum*BoxSize^2,1);
    5*ones(SameSampleNum*BoxSize^2,1);6*ones(SameSampleNum*BoxSize^2,1)];

%% Resize or cropped image
if crop_image == 1                                                                                                         % ¡Ú¡Ú¡Ú ¼³Á¤¿ä
cropcube = cropcol(setcube, crop_num, S);                                               
setcube=cropcube;
end

%% Extract training DB : 'X'
showHSI(setcube, selected_bands, LearnFigNum);
[X, cp, rp] = TrainDB(setcube, ClassNum, BoxSize, LearnFigNum, ColorSet, SameSampleNum);

%% Classification by Partial Least Square
trainDB_num = SameSampleNum*ClassNum*(BoxSize^2);                           % the total number of train samples
Y_PLS = kron(eye(ClassNum),ones((SameSampleNum*(BoxSize^2)),1));            % Labeling each class
% normalization X and Y
xmean = mean(X); xstd = std(X); ymean = mean(Y_PLS); ystd = std(Y_PLS);
X_norm = (X - xmean(ones(trainDB_num,1),:))./xstd(ones(trainDB_num,1),:);                        
Y_norm = (Y_PLS - ymean(ones(trainDB_num,1),:))./ystd(ones(trainDB_num,1),:);      
% Tolerance for 90 percent score
tol = (1-0.9) * 25 * 4;                                                     
% Perform PLS
[T,P,U,Q,B,W] = pls(X_norm, Y_norm, tol);
Result_PLS = PLSresult(setcube, X, Y_PLS, P, B, Q);
% Display result
figure(1100); imagesc(Result_PLS); set(gcf,'color','w');
title('Classification by PLS-DA','fontsize',14); colorbar; colormap(classmap); 

%% Classification by Multi-class Adaboost
Y_Ada = Y;
Mdl = fitensemble(X,Y_Ada,'AdaBoostM2',BoxSize^2,'Tree');

Predict = predict(Mdl,X);
figure(1210); bar(Predict); title('Predict bar'); xlabel('the number of samples'); ylabel('labeling number');

Result_Adaboost = Adaboostresult(setcube, Mdl);
figure(1200); imagesc(Result_Adaboost); set(gcf,'color','w');
title('Classification by Adaboost','fontsize',14); colorbar;colormap(classmap);

%% Classification by Mult-class SVM
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




















