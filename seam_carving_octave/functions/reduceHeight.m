function [Image, ImagesOut, EnergyMaps, minEnergyMaps] = reduceHeight(ImageIn, numPixels=1, energy_function=1, create_animations=0)

  if (ndims(ImageIn)==1)
    ImageIn = repmat(ImageIn, [1 1 3]);
  endif

  Image = im2double(ImageIn);

  [height, width, ~] = size(Image);

  ImagesOut     = zeros([height, width, 3, numPixels+1]);
  ImagesOut(:,:,:,1) = Image;
  EnergyMaps    = zeros([height, width, 3, numPixels]);
  minEnergyMaps = zeros([height, width, 3, numPixels]);

  target = height - numPixels + 1;

  for current_height = height:-1:target

    Energy = energy(rgb2gray(Image), energy_function);

    minEnergy = minHorizontalEnergyMap(Energy);

    seam = findOptimalHorizontalSeam(minEnergy);

    if (create_animations)
      ImagesOut(1:current_height,1:width,:,2+height-current_height) = overlayHorizontalSeam(Image, seam);

      EnergyMaps(1:current_height,1:width,:,1+height-current_height) = overlayHorizontalSeam(Energy, seam);

      m=min(min(minEnergy));
      M=max(max(minEnergy));
      minEnergyMaps(1:current_height,1:width,:,1+height-current_height) = overlayHorizontalSeam((minEnergy-m)/(M-m), seam);
    endif

    i = logical(1:current_height==seam(:))';
    Image([i i i]) = [];
    Image = reshape(Image, current_height-1, width, 3);

  endfor

  if (ndims(ImageIn)==1)
    ImagesOut = ImagesOut(:,:,1,:);
  endif

  Image         = im2uint8(Image);
  ImagesOut     = im2uint8(ImagesOut);
  EnergyMaps    = im2uint8(EnergyMaps);
  minEnergyMaps = im2uint8(minEnergyMaps);

endfunction
