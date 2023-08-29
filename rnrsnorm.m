function NormCube = rnrsnorm(cube)
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% rnrsnorm(cube) 
% 
% input : Hyperspectral data 
%         ex) cube(row, vertical, col)
%             row : row in image
%             col : col in image
%             vertical : spectral profile
%
% output : Normalized hyperspectral data
%
% Described function :
% Normalized hyperspectral profile in each pixel
% Remove minimum value and rescale maximum value
% 
% Made by Heekang Kim.
% Date is 2016.3.22
% update 2016.04.21 (rescale to one)
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[row vertical col] = size(cube);

NormCube = zeros(row, vertical, col);
for Row = 1:row
    for Col = 1:col
        raw = cube(Row, :, Col);
        if max(raw-min(raw)) == 0
            NormCube(Row, :, Col) = raw;
        else
        NormCube(Row, :, Col) = ((raw-min(raw))/max(raw-min(raw)));
        end
    end
end
end
