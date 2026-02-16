function [Image, ImagesOut, EnergyMaps, minEnergyMaps] = reduceWidth(ImageIn, numPixels=1, energy_function=1, create_animations=0)
  ImageInT = transpose_hxwx3_matrix(ImageIn);
  [ImageT, ImagesOutT, EnergyMaps, minEnergyMaps] = reduceHeight(ImageInT, numPixels, energy_function, create_animations);
  Image = transpose_hxwx3_matrix(ImageT);
  ImagesOut = transpose_hxwx3_matrix(ImagesOutT);
  EnergyMaps = transpose_hxwx3_matrix(EnergyMaps);
  minEnergyMaps = transpose_hxwx3_matrix(minEnergyMaps);
endfunction
