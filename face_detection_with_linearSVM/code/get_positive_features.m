% Starter code prepared by James Hays for CS 143, Brown University
% This function should return all positive training examples (faces) from
% 36x36 images in 'train_path_pos'. Each face should be converted into a
% HoG template according to 'feature_params'. For improved performance, try
% mirroring or warping the positive training examples.

function features_pos = get_positive_features(train_path_pos, feature_params)
% 'train_path_pos' is a string. This directory contains 36x36 images of
%   faces
% 'feature_params' is a struct, with fields
%   feature_params.template_size (probably 36), the number of pixels
%      spanned by each train / test template and
%   feature_params.hog_cell_size (default 6), the number of pixels in each
%      HoG cell. template size should be evenly divisible by hog_cell_size.
%      Smaller HoG cell sizes tend to work better, but they make things
%      slower because the feature dimensionality increases and more
%      importantly the step size of the classifier decreases at test time.


% 'features_pos' is N by D matrix where N is the number of faces and D
% is the template dimensionality, which would be
%   (feature_params.template_size / feature_params.hog_cell_size)^2 * 31
% if you're using the default vl_hog parameters

% Useful functions:
% vl_hog, HOG = VL_HOG(IM, CELLSIZE)
%  http://www.vlfeat.org/matlab/vl_hog.html  (API)
%  http://www.vlfeat.org/overview/hog.html   (Tutorial)
% rgb2gray

images     = readall( imageDatastore(train_path_pos) );
num_images = length(images);

fphcs = feature_params.hog_cell_size; 
template_dimensionality = (feature_params.template_size / feature_params.hog_cell_size)^2 * 31;

features_pos = zeros(template_dimensionality, 2, num_images);

flipped_indices = vl_hog('permutation');

parfor i=1:num_images
    img = im2single( images{i} );
    
    hog = vl_hog(img, fphcs);
    mirrored_hog = flip( hog(:,:,flipped_indices), 2 );

    features_pos(:,:,i) = [ ...
        reshape( hog, template_dimensionality, 1 ) ...
        reshape( mirrored_hog, template_dimensionality, 1 )
    ];
end

features_pos = reshape(features_pos, template_dimensionality, 2*num_images)';