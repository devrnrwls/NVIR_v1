function filterCube = medcube(cube, n)
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% medcube(cube, n)
%
% input : Hyperspectral data 
%         ex) cube(row, vertical, col)
%             row : row in image
%             col : col in image
%             vertical : spectral profile
%          n : filter size
%
% output : Normalized hyperspectral data
%
% Described function :
% 각 픽셀의 밴드별 Salt and Pepper 잡음 제거하기 위한 Median 필터 적용
%
% 
% Made by Heekang Kim.
% Date is 2016.4.13
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if rem(n,2) == 0
    error('it must be odd number');
end

[row vertical col] = size(cube);
newcube = zeros(row, vertical, col);

for ROW = 1:row
    for COL = 1:col
        
            raw = cube(ROW, :, COL);
            newcube(ROW, :, COL) = medfilt1(raw, n);

    end
    disp(ROW*100/row)
end

filterCube = newcube;

end