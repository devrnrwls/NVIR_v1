function showHSI(cube, selected_bands, fignum)

r_band = selected_bands(3);
g_band = selected_bands(2);
b_band = selected_bands(1);

R=squeeze(cube(:,r_band,:));                              % ROI ������ ������ ���� ���� (111��° ����� ���ð� ���� R�� ����)
G=squeeze(cube(:,g_band,:));                              % ROI ������ �ʷϻ� ���� ���� (75 ��° ����� ���ð� ���� G�� ����)
B=squeeze(cube(:,b_band,:));                              % ROI ������ �Ķ��� ���� ���� (38 ��° ����� ���ð� ���� B�� ����)

S(:,:,1)=R/max(R(:))*255;                                % ������ ������ ȭ�鿡 ��Ÿ�� �� �ְ� ����
S(:,:,2)=G/max(G(:))*255;                                % �ʷϻ� ������ ȭ�鿡 ��Ÿ�� �� �ְ� ����
S(:,:,3)=B/max(B(:))*255;                                % �Ķ��� ������ ȭ�鿡 ��Ÿ�� �� �ְ� ����

figure(fignum); clf, set(gcf,'color','w');                % �� �ռ�             
imagesc(uint8(S)); axis image; truesize;                   
title('Recomposition image from HSI to RGB', 'fontsize',14); xlabel('Lines','fontsize',13); ylabel('Samples', 'fontsize',13); axis off; box off;
