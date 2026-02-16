function [Energy] = energy(GrayImageIn, p)
  [Gx, Gy] = imgradientxy(GrayImageIn);
  if (p >= 1)
    Energy = (abs(Gx).^p + abs(Gy).^p).^(1/p);    # L_p norm
  elseif (p == 0)
    Energy = max(abs(Gx), abs(Gy));               # L_infinity norm
  else
    error("wrong energy function")
  endif
endfunction
