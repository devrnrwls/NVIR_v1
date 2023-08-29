function Result_image = PLSresult(cube, X, Y, P, B, Q)
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PLSresult(cube, X, Y, P, B, Q)
%
% input : 
%       cube : hyperspectral image 3D cube
%       X : set of train DB
%       Y : set of labeling
%       P : Extraction feature vector from principal component axis vertor of X
%       B : Regression coefficients ( U = TB )
%       Q : Principal component axis vector of Y
%
% output :
%       Result_image : Result of Classification
%
% Made by Heekang Kim.
% ver1 : 2016.04.29
% ver2 : 2016.06.10 (Description added, Dead code elimination)
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[row, vertical, col] = size(cube);
Result_image = zeros(row, col);

ymean = mean(Y);                                                           
ystd = std(Y);  

xmean = mean(X);
xstd = std(X);


for ROW = 1:row
    disp(ROW/row);
    for COL = 1:col
        sample = cube(ROW, :, COL);
        sample_norm = (sample-xmean)./xstd;
        Y0 = sample_norm*(P*B*Q');
        Y_norm = Y0 .* ystd + ymean;
        [dum, classid] = min(abs(Y_norm -1), [], 2);
        Result_image(ROW, COL) = classid;
    end
end


end







