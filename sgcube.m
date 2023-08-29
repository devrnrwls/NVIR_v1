function filterCube = sgcube(cube, k, f)

if rem(f,2) == 0
    error('it must be odd number');
end

[row vertical col] = size(cube);
newcube = zeros(row, vertical, col);

for ROW = 1:row
    for COL = 1:col
       
        raw = cube(ROW, :, COL);
        sgolay_raw = sgolayfilt(raw, k, f);
        newcube(ROW, :, COL) = sgolay_raw;
        
    end
        disp(ROW*100/row)
end



filterCube = newcube;
end
