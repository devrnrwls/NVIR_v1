clear;
clc;
close all;

%% Load HSI raw data
    [cube, Wavelength, rct5, selected_bands S]=readDGISTVNIRHSI(0);
    [row, vertical, col] = size(cube);
    

    %% base set !!����� �̰͸� �����ϸ� �ȴ�!
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % ��������;
    % � ��ó�� ����� �����ϰڴ°� *��ó�� ����
    setcube = cube;                                                   % �ڡڡ� ������
    %-------------------------------------------------------------------------
    % ClassNum : �н��� ���� �ڡڡڡڡڡڡڡڡڡڡڡڡڡڡڡڡڡڡڡڡڡڡڡڡڡڡڡڡ�
    ClassNum = 3;                                                               % �ڡڡ� ������
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

    
    % �н� ������ ���
    sample = X(1:numeachsample*ClassNum, :);
    
    ave1 = mean(sample1);
    ave2 = mean(sample2);
    ave3 = mean(sample3);

    
    std1 = std(sample1);
    std2 = std(sample2);
    std3 = std(sample3);

    
    
    % �н� ������ ���, ǥ������
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

    % NCI ��� ��
    
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
    %% Classification by Multi-class SVM�� �̿��� �з��� �н�
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
    %% Classification by Multi-class SVM�� �̿��� �з��� �н�
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

