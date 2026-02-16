function [EnergyOut] = minVerticalEnergyMap(Energy)
  EnergyOut = minHorizontalEnergyMap(Energy')';
endfunction

