% Starter code prepared by James Hays for CS 143, Brown University
% This function should return negative training examples (non-faces) from
% any images in 'non_face_scn_path'. Images should be converted to
% grayscale, because the positive training data is only available in
% grayscale. For best performance, you should sample random negative
% examples at multiple scales.

function features_neg = get_random_negative_features(non_face_scn_path, feature_params, num_samples)
% 'non_face_scn_path' is a string. This directory contains many images
%   which have no faces in them.
% 'feature_params' is a struct, with fields
%   feature_params.template_size (probably 36), the number of pixels
%      spanned by each train / test template and
%   feature_params.hog_cell_size (default 6), the number of pixels in each
%      HoG cell. template size should be evenly divisible by hog_cell_size.
%      Smaller HoG cell sizes tend to work better, but they make things
%      slower because the feature dimensionality increases and more
%      importantly the step size of the classifier decreases at test time.
% 'num_samples' is the number of random negatives to be mined, it's not
%   important for the function to find exactly 'num_samples' non-face
%   features, e.g. you might try to sample some number from each image, but
%   some images might be too small to find enough.

% 'features_neg' is N by D matrix where N is the number of non-faces and D
% is the template dimensionality, which would be
%   (feature_params.template_size / feature_params.hog_cell_size)^2 * 31
% if you're using the default vl_hog parameters

% Useful functions:
% vl_hog, HOG = VL_HOG(IM, CELLSIZE)
%  http://www.vlfeat.org/matlab/vl_hog.html  (API)
%  http://www.vlfeat.org/overview/hog.html   (Tutorial)
% rgb2gray

images     = readall( imageDatastore(non_face_scn_path) );
num_images = length(images);

fpts  = feature_params.template_size;
fphcs = feature_params.hog_cell_size;
window_size = fpts/fphcs;

template_dimensionality = window_size^2 * 31;

samples_from_each_image = ceil( num_samples/num_images );

features_neg = zeros( num_images*samples_from_each_image, template_dimensionality );

for i=1:num_images
    img = im2single( rgb2gray( images{i} ) );

    hog      = vl_hog(img, fphcs);
    [h,w,~]  = size(hog);
    flat_hog = reshape(hog, [h*w 31] );
    
    window_indices = (0:window_size-1)*h + (1:window_size)';
    x = randi([0, h-window_size], 1, 1, samples_from_each_image);
    y = randi([0, w-window_size], 1, 1, samples_from_each_image);
    random_windows = window_indices + x + h*y;

    samples = permute( ...
        reshape( ...
            flat_hog(random_windows,:), ...
            window_size^2, ...
            samples_from_each_image, ...
            31 ...
         ), ...
         [2 1 3] ...
    );

    features_neg( (1:samples_from_each_image) + (i-1)*samples_from_each_image, : ) = ...
        reshape(samples, [samples_from_each_image template_dimensionality]);
end