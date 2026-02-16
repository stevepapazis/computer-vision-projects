function [Seam] = findOptimalVerticalSeam(minimumEnergyMap)
  Seam = findOptimalHorizontalSeam(minimumEnergyMap')';
endfunction
