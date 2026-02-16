function [EnergyOut] = minHorizontalEnergyMap(Energy)

  M = Energy;

  l = rows(M);

  for i = 2:columns(M)

    M(1,i) += min( M(1,i-1), M(2,i-1) );
    M(l,i) += min( M(l-1,i-1), M(l,i-1) );

    r = 2:l-1;
    M(r,i) += min(  min( M(r-1,i-1), M(r,i-1) ), M(r+1,i-1)  );

  endfor

  EnergyOut = M;

endfunction

