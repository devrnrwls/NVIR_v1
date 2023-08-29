function showHSI(cube, selected_bands, fignum)

r_band = selected_bands(3);
g_band = selected_bands(2);
b_band = selected_bands(1);

R=squeeze(cube(:,r_band,:));                              % ROI 영역의 빨간색 성분 추출 (111번째 밴드의 샘플과 라인 R에 저장)
G=squeeze(cube(:,g_band,:));                              % ROI 영역의 초록색 성분 추출 (75 번째 밴드의 샘플과 라인 G에 저장)
B=squeeze(cube(:,b_band,:));                              % ROI 영역의 파란색 성분 추출 (38 번째 밴드의 샘플과 라인 B에 저장)

S(:,:,1)=R/max(R(:))*255;                                % 빨간색 성분을 화면에 나타날 수 있게 설정
S(:,:,2)=G/max(G(:))*255;                                % 초록색 성분을 화면에 나타날 수 있게 설정
S(:,:,3)=B/max(B(:))*255;                                % 파란색 성분을 화면에 나타날 수 있게 설정

figure(fignum); clf, set(gcf,'color','w');                % 색 합성             
imagesc(uint8(S)); axis image; truesize;                   
title('Recomposition image from HSI to RGB', 'fontsize',14); xlabel('Lines','fontsize',13); ylabel('Samples', 'fontsize',13); axis off; box off;
