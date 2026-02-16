function [ImageOut] = transpose_hxwx3_matrix(ImageIn)
  ImageOut = permute(ImageIn, [2,1,3:ndims(ImageIn)]);
endfunction
