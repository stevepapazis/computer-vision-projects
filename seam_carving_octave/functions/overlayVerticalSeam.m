function [ImageOut] = overlayVerticalSeam(ImageIn, Seam, color=[1,0,0])

  ImageOut = transpose_hxwx3_matrix(
    overlayHorizontalSeam(
      transpose_hxwx3_matrix(ImageIn),
      Seam'
    )
  );

endfunction
