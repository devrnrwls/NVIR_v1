clear;
clc;
close all;

%% Load HSI raw data
    [cube, Wavelength, rct5, selected_bands S]=readDGISTVNIRHSI(0);
    [row, vertical, col] = size(cube);
    

    %% base set !!당신은 이것만 설정하면 된다!
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % 변수설명;
    % 어떤 전처리 결과를 선택하겠는가 *전처리 선택
    setcube = cube;                                                   % ★★★ 설정요
    %-------------------------------------------------------------------------
    % ClassNum : 학습할 개수 ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    ClassNum = 3;                                                               % ★★★ 설정요
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
    

    
    %% Extract training DB
    showHSI(setcube, selected_bands, 100);
    N= ClassNum;
    Box= Boxsize;
    L= learing_num;
    [X, cp, rp] = PLSlearn(setcube, N, Box, fignum, colorset, L);
    
    Y=[ones(100,1);2*ones(100,1);3*ones(100,1)];
    %Y=[ones(100,1);2*ones(100,1);3*ones(100,1);4*ones(100,1);5*ones(100,1);6*ones(100,1)]; % label
    
    
    %% Calculation each band subtraction to find best band set
    numeachsample=(Boxsize^2);
    
    sample1 = X(1:numeachsample, :);
    sample2 = X(numeachsample+1:numeachsample*2, :);
    sample3 = X(numeachsample*2+1:numeachsample*3, :);

    
    % 학습 데이터 행렬
    sample = X(1:numeachsample*ClassNum, :);
    
    ave1 = mean(sample1);
    ave2 = mean(sample2);
    ave3 = mean(sample3);

    
    std1 = std(sample1);
    std2 = std(sample2);
    std3 = std(sample3);

    
    
    % 학습 데이터 평균, 표준편차
    ave_set = [ave1; ave2; ave3];
    std_set = [std1; std2; std3];
    combos = [1, 2;
                1, 3;
                2, 3]
%    combntns(1:3, 2);
    
  [ temp1, numBand, temp2]  = size(setcube);
     
    for bands = 1:numBand

        for i = 1:size(combos,1)
            indx1= combos(i,1);
            indx2= combos(i,2);
            %sub_DN(i) = abs( ave_set(indx1,bands) - ave_set(indx2,bands) ) / (std_set(indx1, bands)+std_set(indx2, bands));    
            sub_DN(i) = abs( ave_set(indx1,bands) - ave_set(indx2,bands) );
        end

        NCI(bands) = min(sub_DN);
        %max_subDN(bands) = max(sub_DN);
    end

    %figure(200); plot(Wavelength, NCI, 'b'); hold on;
    %ylabel('Nearest Class Interval (NCI)'); xlabel('Wavelength (nm)');set(gcf,'color','w');
    
    %figure(210); plot(Wavelength, max_subDN, 'b');

    % NCI 계산 끝
    
%% find candidate bands by local max (findpeak)
    max_NCI = max(NCI);
    %pksTH = 0.1*max_NCI;
    
    %Wavelength = Wavelength(1:142);

    [pksLocal,locsLocal] = findpeaks(NCI,'MINPEAKHEIGHT',50, 'MINPEAKDISTANCE',5);
    figure(210); plot(Wavelength, NCI, 'b'); hold on; plot(Wavelength(locsLocal), pksLocal, 'ro');
    ylabel('Nearest Class Interval (NCI)'); xlabel('Wavelength (nm)');set(gcf,'color','w');
    % Set cube with band selection
    BandSelection_cube1 = zeros(row, size(locsLocal, 2), col);
    
    for band = 1:size(locsLocal, 2)
        BandSelection_cube1(:, band, :) = setcube(:, locsLocal(band), :);
    end
    setcube2 = BandSelection_cube1;
    

 %% Extract selected band from X1 DB : 'X2'
X2 = zeros(size(X, 1), size(locsLocal, 2));

for band0 = 1:size(locsLocal, 2)
   X2(:, band0) = X(:, locsLocal(band0));
end   
    

%% Ground truth generation (FULL BAND)
if 1
    maxNo=vertical;
    bindx=1:maxNo;
    XM=X(:,bindx);
    % fast SVM-SAM
    %% Classification by Multi-class SVM을 이용한 분류기 학습
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
    
    % Test for the input
    xtest=XM;
    TrueID=Y;
    
    [ypred,maxi] = svmmultival(xtest,xsup,w,b,nbsv,kernel,kerneloption);
    figure(200), plot(ypred);
    pause(1);
    
    % test
    disp(['Test using trained DB']);
    [row_2, vertical_2, col_2] = size(setcube);
    result=zeros(row_2,col_2);
    % transform the cube to 2D array
    Cube2D = permute(setcube,[1 3 2]);
    Cube2D = reshape(Cube2D,[],size(setcube,2),1);
    Cube2DbandSelect=Cube2D(:,bindx);
    tic
    [ypred,maxi] = svmmultival(Cube2DbandSelect,xsup,w,b,nbsv,kernel,kerneloption);
    toc
    result=reshape(ypred,[row_2 col_2]);
    classmap = colorset(1:N,:);
    figure(200); imagesc(result); set(gcf,'color','w'); title('Classification by SVM','fontsize',14); colorbar;colormap(classmap);
    save GT result
    
end


%% Ground truth generation (SELECTED BAND)
if 1
    [temp maxNo]= size(locsLocal);
    bindx2=1:maxNo;
    %XM2=X2(:,bindx);
    % fast SVM-SAM
    %% Classification by Multi-class SVM을 이용한 분류기 학습
    tic
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
    [xsup2,w2,b2,nbsv2]=svmmulticlassoneagainstall(X2,Y,nbclass,c,lambda,kernel,kerneloption,verbose);
    
    % Test for the input
    xtest=X2;
    TrueID=Y;
    
    [ypred,maxi] = svmmultival(xtest,xsup2,w2,b2,nbsv2,kernel,kerneloption);
    figure(200), plot(ypred);
    pause(1);
    
    % test
    disp(['Test using trained DB']);
    [row_2, vertical_2, col_2] = size(setcube2);
    result=zeros(row_2,col_2);
    % transform the cube to 2D array
    Cube2D2 = permute(setcube2,[1 3 2]);
    Cube2D2 = reshape(Cube2D2,[],size(setcube2,2),1);
    Cube2DbandSelect2=Cube2D2(:,bindx2);
    tic
    [ypred,maxi] = svmmultival(Cube2DbandSelect2,xsup2,w2,b2,nbsv2,kernel,kerneloption);
    toc
    result=reshape(ypred,[row_2 col_2]);
    classmap = colorset(1:N,:);
    figure(201); imagesc(result); set(gcf,'color','w'); title('Classification by SVM','fontsize',14); colorbar;colormap(classmap);
    save GT result
    toc
end

