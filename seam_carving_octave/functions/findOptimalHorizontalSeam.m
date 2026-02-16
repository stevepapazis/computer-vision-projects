function [Seam] = findOptimalHorizontalSeam(minimumEnergyMap)
  E = minimumEnergyMap;

  [height, width] = size(E);
  Seam = zeros(1, width);

  [~, x] = min(E(:, width));
  Seam(width) = x;

  for y = width-1:-1:1
    if (x == 1)
      [~, x] = min(E(1:2, y));
    elseif (x == height)
      [~, dx] = min(E(height-1:height, y)); # dx is 1 or 2
      x += dx-2;
    else
      [~, dx] = min(E(x-1:x+1, y));         # dx is 1, 2 or 3
      x += dx-2;
    endif

    Seam(y) = x;

  endfor
endfunction
