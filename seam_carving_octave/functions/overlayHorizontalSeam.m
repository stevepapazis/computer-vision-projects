function [ImageOut] = overlayHorizontalSeam(ImageIn, Seam, color=[1,0,0])

  if (ndims(ImageIn) < 3)
    ImageOut = cat(3, ImageIn, ImageIn, ImageIn);
  else
    ImageOut = ImageIn;
  endif

  [height, width, ~] = size(ImageOut);

  idx = sub2ind([height, width], Seam, 1:width);

  ImageOut([idx; idx+height*width; idx+2*height*width]) = repmat(color(:), 1, width);

endfunction