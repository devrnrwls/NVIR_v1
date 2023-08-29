function [cube2, Wavelength, rct, selectedBand, S2]=readNIRHSI(fnameHD, fnameRaw, fignum)


r_band = 30;
g_band = 60;
b_band = 90;
selectedBand = [r_band, g_band, b_band];

%% load header information
fid_header=fopen(fnameHD,'r');                          % fnameHD 파일을 열어준다음, 'r' reading 파일을 읽어 fid_header에 저장
for i=1:3                                               % A_vector가 dummy값 뜻함
    A_vector=fgets(fid_header);                          % 10줄을 스킵하겠다는 뜻
end
for k=1:3
    A_vector=fgets(fid_header);
    num(k,:)=sscanf(A_vector,'%*s %*s %d', [1, inf]);   % sscanf:문자열을 주어진 포맷에 맞춰 다시 읽는 역할
end

samples = num(1);           % 10 + 1번째 줄 샘플 값
lines   = num(2);           % 10 + 2번째 줄 라인 값
bands   = num(3);           % 10 + 3번째 줄 밴드 값

for i=1:7                                              % 이후 20번째 줄 스킵 (34번부터) : 파장 데이터 값
    A_vector=fgets(fid_header)  ;                         
end


Wave = zeros(26, 8);
disp('check0')

%load wavelength data           
for i=1:25                                              % 1부터 258(밴드수)까지 34~292번 줄 까지
    B_vector=fgets(fid_header);                         % B_vector 값에 밴드 값을 저장함(문자인식)
    
    for r = 1:8
        data = str2num(B_vector);
        Wave(i,r)=data(r);                    % B_vector 값을 숫자로 변환하여 wavelength(i)에 저장
    end
end
HDR_Row = bands/8;
disp('check1')
B_vector=fgets(fid_header);  
data = str2num(B_vector(1:63));
Wave(HDR_Row, :) = data(:);
Wave = Wave';

Wavelength = zeros(1, bands);
Wavelength(:) = Wave(:);
    
fclose(fid_header);                                     % fid_header 닫기

%% load raw file data
fid=fopen(fnameRaw,'r');                                % fnameRaw 파일을 열어준다음, 'r' reading 파일을 읽어 fid에 저장
numTotal=samples*bands*lines;                           % 총 숫자는 샘풀값 * 밴드값 * 라인값 임
data=fread(fid,numTotal,'uint16');                      % 저장한 파일 값을 읽어들여 총 숫자로 
fclose(fid);

cube=reshape(data, samples, bands, lines);              % reshape : data 변수를 샘플(행), 밴드(열), 라인(3차원)으로 재구성

clear data;
%% Display 3D cube data
figure(fignum+1); set(gcf,'color','w');                 % figure1 -> 각 스펙트럼 밴드별 사진을 올림
for i=1:1:bands % image up-down fliping                 % 밴드 수만큼 반복
    Wavelength(i)                                       % wavelength 값
    R1=squeeze(cube(:,i,:));                             % squeeze: 차원을 낮춤
%     R=flipud(R);                                        % filpud(R): R의 상단 행과 하단행 교체- 왜이런지는..?-> 영상이 뒤집혀진가봄 (영상의 좌표기준)
    cube(:,i,:)=R1;                                      % cube에 상단 행 하단 행 교체
end
for i=1:2:bands                                        % 10개 밴드를 반복해서
    Wavelength(i)                                       % wavelength 값
    R1=squeeze(cube(:,i,:));                             % 각 밴드
    figure(fignum+1); imagesc(R1); text(20,40,[num2str(Wavelength(i)) 'nm'], 'fontsize',13,'color','w'); axis off; %truesize
    pause(0.01);                                         % 0.1초 뒤에 다음 밴드 사진 보여줌
end

figure(fignum+1); title('Spatial image per band', 'fontsize',14); xlabel('Lines','fontsize',13); ylabel('Samples', 'fontsize',13); axis off; 
                                                        % title: 그림 창의 제목/ x라벨의 이름 및 크기 13
                                                        % y라벨 이름 및 크기/ axis off : 측 표시값 제거 
                                                        
R=squeeze(cube(:,r_band,:));                            % 빨간색 성분 지정 (111번째 밴드의 샘플과 라인 R에 저장)  
G=squeeze(cube(:,g_band,:));                            % 초록색 성분 추출 (75 번째 밴드의 샘플과 라인 G에 저장)
B=squeeze(cube(:,b_band,:));                            % 파란색 성분 추출 (38 번째 밴드의 샘플과 라인 B에 저장)

S(:,:,1)=R/max(R(:))*255;                               % 빨간색 성분을 화면에 나타날 수 있게 설정
S(:,:,2)=G/max(G(:))*255;                               % 초록색 성분을 화면에 나타날 수 있게 설정
S(:,:,3)=B/max(B(:))*255;                               % 파란색 성분을 화면에 나타날 수 있게 설정

figure(fignum+2); clf, set(gcf,'color','w'); imagesc(uint8(S)); %axis image; truesize, % 색 합성으로 실제 사진으로 합성
    % clf :현재 떠있는 창 제거, set(gcf):배경을 흰색, imagesc:256단계의 gray scale 이미지보기(RGB)

title('Synthesized image', 'fontsize',14); xlabel('Lines','fontsize',13); ylabel('Samples', 'fontsize',13); axis off;
axis off;

%imwrite(uint8(S), 'S.bmp');

%% Cropped cube
disp('Select ROI');
rct=round(getrect());                                       % round : 반올림 함수, getrect: 사각형 포인트 시작점 끝점 생성 [x시작 y시작 x끝 y끝] 형태임
cube2=cube(rct(2):rct(2)+rct(4),:,rct(1):rct(1)+rct(3));    % 복원된 영상을 기준으로 샘플은 y축, 라인은 x축임

clear cube;

R2=squeeze(cube2(:,r_band,:));                              % ROI 영역의 빨간색 성분 추출 (111번째 밴드의 샘플과 라인 R에 저장)
G2=squeeze(cube2(:,g_band,:));                              % ROI 영역의 초록색 성분 추출 (75 번째 밴드의 샘플과 라인 G에 저장)
B2=squeeze(cube2(:,b_band,:));                              % ROI 영역의 파란색 성분 추출 (38 번째 밴드의 샘플과 라인 B에 저장)

S2(:,:,1)=R2/max(R2(:))*255;                                % 빨간색 성분을 화면에 나타날 수 있게 설정
S2(:,:,2)=G2/max(G2(:))*255;                                % 초록색 성분을 화면에 나타날 수 있게 설정
S2(:,:,3)=B2/max(B2(:))*255;                                % 파란색 성분을 화면에 나타날 수 있게 설정

figure(fignum+3); clf, set(gcf,'color','w');                % 색 합성             
imagesc(uint8(S2)); axis image; truesize,                   
title('Cropped Synthesized image', 'fontsize',14); xlabel('Lines','fontsize',13); ylabel('Samples', 'fontsize',13); axis off; box off;

end