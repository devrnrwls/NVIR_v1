function [X, cp, rp] = PLSlearn(cube, num, boxsize, fignum, colorset, learing_num)
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PLSlearn(cube, num, boxsize)
%
% input : 
%
%
%
% output :
%
% Described function :
%
%
% Made by Heekang Kim.
% Date is 2016.04.29
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[row, vertical, col] = size(cube);

N = num;
Box = boxsize;
cp = zeros(1, num);
rp = zeros(1, num);
raw_data_set = [];

for i=1:N
    
    for sample = 1:learing_num
        figure(fignum); hold on;
        [cp(i) rp(i)]=(getpts); cp(i)=round(cp(i));rp(i)=round(rp(i));                       
        raw=cube((rp(i)):(rp(i))+Box-1, : ,(cp(i)):(cp(i))+Box-1); %100:30:300            
        
        cdata = colorset(i,:);
        rectangle('Position',[cp(i) rp(i) Box Box],'EdgeColor',cdata);          % 영상에 사각형 남기기
        
        % re-arrange the cropped cube matrix
        [row_1, vertical_1, col_1] = size(raw);                                 % 채취한 초분광 데이터 크기 확인f
        raw_data=zeros(row_1*col_1, vertical_1);                                % 2D 영상을 1D 백터로 변환
        for rr = 1 : row_1 
            for cc = 1 : col_1
                raw1=raw(rr,:, cc);
                raw_data((rr-1)*col_1+cc,:) = raw1;    
            end
        end
        raw_data_set=[ raw_data_set; raw_data];
    end
    Xtot{i}=raw_data_set(:,:);                                                       % 1D로 된 영상 벡터와, 밴드벡터
    raw_data_set = [];
end

idxTrain = 1:(Box^2)*learing_num;
if N == 2
    x1=Xtot{1}; x2=Xtot{2};
    X = [x1(idxTrain,:);x2(idxTrain,:)];
elseif N == 3
    x1=Xtot{1}; x2=Xtot{2};x3=Xtot{3};  
    X = [x1(idxTrain,:);x2(idxTrain,:);x3(idxTrain,:)];
elseif N == 4
    x1=Xtot{1}; x2=Xtot{2};x3=Xtot{3}; x4=Xtot{4};
    X = [x1(idxTrain,:);x2(idxTrain,:);x3(idxTrain,:) ; x4(idxTrain,:)];
elseif N == 5
    x1=Xtot{1}; x2=Xtot{2};x3=Xtot{3}; x4=Xtot{4}; x5=Xtot{5};                                 
    X = [x1(idxTrain,:);x2(idxTrain,:);x3(idxTrain,:) ; x4(idxTrain,:); x5(idxTrain,:)];
elseif N == 6
    x1=Xtot{1}; x2=Xtot{2};x3=Xtot{3}; x4=Xtot{4}; x5=Xtot{5}; x6=Xtot{6};                          
    X = [x1(idxTrain,:);x2(idxTrain,:);x3(idxTrain,:) ; x4(idxTrain,:); x5(idxTrain,:);x6(idxTrain,:)];
end










