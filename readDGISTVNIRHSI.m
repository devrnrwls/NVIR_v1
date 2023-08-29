function [cube2, Wavelength, rct, selectedBand, S2]=readDGISTVNIRHSI(fignum)
% load HSI data
%% load HSI data
[fname, path] = uigetfile('D:\.raw');

%%C:\Users\Seongho\Desktop\emptyname_2019-09-05_18-22-49\emptyname_2019-09-05_18-22-49\capture\.raw


fnameHD = [path fname(1:length(fname)-4) '.hdr'];                           % ������� �̸�
fnameRaw = [path fname];                                                    % Raw ���� �̸�

%% load header information
fid_header=fopen(fnameHD,'r');                          % fnameHD ������ �����ش���, 'r' reading ������ �о� fid_header�� ����
for i=1:10                                              % A_vector�� dummy�� ����
    A_vector=fgets(fid_header);                          % 10���� ��ŵ�ϰڴٴ� ��
end
for k=1:3
    A_vector=fgets(fid_header);
    num(k,:)=sscanf(A_vector,'%*s %*s %d', [1, inf]);   % sscanf:���ڿ��� �־��� ���˿� ���� �ٽ� �д� ����
end

samples = num(1);           % 10 + 1��° �� ���� ��
bands   = num(2);           % 10 + 2��° �� ���� ��
lines = num(3);           % 10 + 3��° �� ��� ��

for i=1:9                                            % ���� 20��° �� ��ŵ (34������) : ���� ������ ��
    A_vector=fgets(fid_header);                           
end

def_bands=fgets(fid_header); % 14��° �� : default band �� : { 111, 75, 38 } -> R, G, B ��
r_band=str2num(def_bands(18:19));                       % 111
g_band=str2num(def_bands(22:23));                       % 75
b_band=str2num(def_bands(26:27));                       % 38 
selectedBand=[b_band g_band r_band];                     % ��� ������ [ 38 75 111 ] ���� ������

for i=1:25                                 % ���� 20��° �� ��ŵ (34������) : ���� ������ ��
    A_vector=fgets(fid_header);                           
end

% %load wavelength data
% bandgap = (1023.7 - 356.1)/(bands-1);
% Wavelength = 356.1:bandgap:1023.7;

           
for i=1:bands                                           % 1���� 258(����)���� 34~292�� �� ����
    B_vector=fgets(fid_header);                         % B_vector ���� ��� ���� ������(�����ν�)
    Wavelength(i)=str2num(B_vector);                    % B_vector ���� ���ڷ� ��ȯ�Ͽ� wavelength(i)�� ����
end
fclose(fid_header);                                     % fid_header �ݱ�

%default bands;

%selectedBand=[r_band g_band b_band];                    % ��� ������ [ 111 75 38 ] ���� ������


%% load raw file data
fid=fopen(fnameRaw,'r');                                % fnameRaw ������ �����ش���, 'r' reading ������ �о� fid�� ����
numTotal=samples*bands*lines;                           % �� ���ڴ� ��Ǯ�� * ��尪 * ���ΰ� ��
data=fread(fid,numTotal,'uint16');                      % ������ ���� ���� �о�鿩 �� ���ڷ� 
fclose(fid);

cube=reshape(data, samples, bands, lines);              % reshape : data ������ ����(��), ���(��), ����(3����)���� �籸��

clear data;
%% Display 3D cube data
figure(fignum+1); set(gcf,'color','w');                 % figure1 -> �� ����Ʈ�� ��庰 ������ �ø�
for i=1:1:bands % image up-down fliping                 % ��� ����ŭ �ݺ�
    Wavelength(i)                                       % wavelength ��
    R=squeeze(cube(:,i,:));                             % squeeze: ������ ����
    R=flipud(R);                                        % filpud(R): R�� ù��° ��� �������� ��ü- ���̷�����..?
    cube(:,i,:)=R;                                      % cube�� 1����� �������� ��ü�� ������ �ٽ� ����
end
for i=1:10:bands                                        % 10�� ��带 �ݺ��ؼ�
    Wavelength(i)                                       % wavelength ��
    R=squeeze(cube(:,i,:));                             % �� ���
    figure(fignum+1); imagesc(R); text(20,40,[num2str(Wavelength(i)) 'nm'], 'fontsize',13,'color','w'); axis off; %truesize
    pause(0.005);                                         % 0.1�� �ڿ� ���� ��� ���� ������
end

figure(fignum+1); title('Spatial image per band', 'fontsize',14); xlabel('Lines','fontsize',13); ylabel('Samples', 'fontsize',13); axis off; 
                                                        % title: �׸� â�� ����/ x���� �̸� �� ũ�� 13
                                                        % y�� �̸� �� ũ��/ axis off : �� ǥ�ð� ���� 

                                                        
R=squeeze(cube(:,r_band,:));                            % ������ ���� ���� (111��° ����� ���ð� ���� R�� ����)  
G=squeeze(cube(:,g_band,:));                            % �ʷϻ� ���� ���� (75 ��° ����� ���ð� ���� G�� ����)
B=squeeze(cube(:,b_band,:));                            % �Ķ��� ���� ���� (38 ��° ����� ���ð� ���� B�� ����)

S(:,:,1)=R/max(R(:))*255;                               % ������ ������ ȭ�鿡 ��Ÿ�� �� �ְ� ����
S(:,:,2)=G/max(G(:))*255;                               % �ʷϻ� ������ ȭ�鿡 ��Ÿ�� �� �ְ� ����
S(:,:,3)=B/max(B(:))*255;                               % �Ķ��� ������ ȭ�鿡 ��Ÿ�� �� �ְ� ����

figure(fignum+2); clf, set(gcf,'color','w'); imagesc(uint8(S)); %axis image; truesize, % �� �ռ����� ���� �������� �ռ�
    % clf :���� ���ִ� â ����, set(gcf):����� ���, imagesc:256�ܰ��� gray scale �̹�������(RGB)

title('Synthesized image', 'fontsize',14); xlabel('Lines','fontsize',13); ylabel('Samples', 'fontsize',13); axis off;
axis off;

%imwrite(uint8(S), 'S.bmp');

%% Cropped cube
disp('Select ROI');
rct=round(getrect());                                       % round : �ݿø� �Լ�, getrect: �簢�� ����Ʈ ������ ���� ���� [x���� y���� x�� y��] ������
cube2=cube(rct(2):rct(2)+rct(4),:,rct(1):rct(1)+rct(3));    % ������ ������ �������� ������ y��, ������ x����

clear cube;

R2=squeeze(cube2(:,r_band,:));                              % ROI ������ ������ ���� ���� (111��° ����� ���ð� ���� R�� ����)
G2=squeeze(cube2(:,g_band,:));                              % ROI ������ �ʷϻ� ���� ���� (75 ��° ����� ���ð� ���� G�� ����)
B2=squeeze(cube2(:,b_band,:));                              % ROI ������ �Ķ��� ���� ���� (38 ��° ����� ���ð� ���� B�� ����)

S2(:,:,1)=R2/max(R2(:))*255;                                % ������ ������ ȭ�鿡 ��Ÿ�� �� �ְ� ����
S2(:,:,2)=G2/max(G2(:))*255;                                % �ʷϻ� ������ ȭ�鿡 ��Ÿ�� �� �ְ� ����
S2(:,:,3)=B2/max(B2(:))*255;                                % �Ķ��� ������ ȭ�鿡 ��Ÿ�� �� �ְ� ����

figure(fignum+3); clf, set(gcf,'color','w');                % �� �ռ�             
imagesc(uint8(S2)); axis image; truesize,                   
title('Cropped Synthesized image', 'fontsize',14); xlabel('Lines','fontsize',13); ylabel('Samples', 'fontsize',13); axis off; box off;

end