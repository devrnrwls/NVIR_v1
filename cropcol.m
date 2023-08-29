function newcube = cropcol(cube, crop_num, S)
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% meannorm(cube) 
%
% input : Hyperspectral data 
%         ex) cube(row, vertical, col)
%             row : row in image
%             col : col in image
%             vertical : spectral profile
%
% output : 
%
% Described function :
% 
% Made by Heekang Kim.
% Date is 2016.04.30
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% direction = 0 : 행 자르기,
% direction = 1 : 열 자르기.
[row, vertical, col] = size(cube);

r = zeros(1, crop_num*2); c = zeros(1, crop_num*2);
length_col = zeros(1, crop_num);

figure(90); imagesc(uint8(S)); axis image; truesize; hold on;

for n = 1:crop_num*2
   [c(n), r(n)]=(getpts);
   c(n)=round(c(n));r(n)=round(r(n));
end

for n = 1:crop_num
    length_col(n) = c(2*n)-c(2*n-1)+1;
    cdata = rand(1,3);
    rectangle('Position', [c(2*n-1), 1, c(2*n)-c(2*n-1), row], 'EdgeColor', cdata);
end


switch(crop_num)
    case 1
        %newcol1 = length_col(1);
        newcube = cube(:, :, c(1):c(2));

    case 2
        newcol1 = length_col(1);
        newcol2 = length_col(1)+length_col(2);
        newcube = zeros(row, vertical);
        newcube(:, :, 1:newcol1) = cube(:, :, c(1):c(2));
        newcube(:, :, (newcol1+1):newcol2) = cube(:, :, c(3):c(4));
        
    case 3
        newcol1 = length_col(1);
        newcol2 = length_col(1)+length_col(2);
        newcol3 = length_col(1)+length_col(2)+length_col(3);
        newcube = zeros(row, vertical);
        newcube(:, :, 1:newcol1) = cube(:, :, c(1):c(2));
        newcube(:, :, newcol1+1:newcol2) = cube(:, :, c(3):c(4));
        newcube(:, :, newcol2+1:newcol3) = cube(:, :, c(5):c(6));
        
        
    case 4
        newcol1 = length_col(1);
        newcol2 = length_col(1)+length_col(2);
        newcol3 = length_col(1)+length_col(2)+length_col(3);
        newcol4 = length_col(1)+length_col(2)+length_col(3)+length_col(4);
        newcube = zeros(row, vertical);
        newcube(:, :, 1:newcol1) = cube(:, :, c(1):c(2));
        newcube(:, :, newcol1+1:newcol2) = cube(:, :, c(3):c(4));
        newcube(:, :, newcol2+1:newcol3) = cube(:, :, c(5):c(6));       
        newcube(:, :, newcol3+1:newcol4) = cube(:, :, c(7):c(8));    
        
end

