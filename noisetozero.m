function NormCube = noisetozero(cube, threshold)
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% pernorm(cube) 
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
% Remove minimum value and rescale maximum value to 100
% 
% Made by Heekang Kim.
% Date is 2015.12.07
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[row vertical col] = size(cube);

NormCube = zeros(row, vertical, col);
for Row = 1:row
    for Col = 1:col
        average_value = mean(cube(Row, :, Col));
        if average_value < threshold
            NormCube(Row, :, Col) = 0;
        
        else
            NormCube(Row, : , Col) = cube(Row, :, Col);
        %NormCube(Row, :, Col) = ((raw-min(raw))/max(raw-min(raw)));
        end
    end
end
