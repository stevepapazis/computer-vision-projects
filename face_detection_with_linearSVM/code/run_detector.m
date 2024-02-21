% Starter code prepared by James Hays for CS 143, Brown University
% This function returns detections on all of the images in a given path.
% You will want to use non-maximum suppression on your detections or your
% performance will be poor (the evaluation counts a duplicate detection as
% wrong). The non-maximum suppression is done on a per-image basis. The
% starter code includes a call to a provided non-max suppression function.
function [bboxes, confidences, image_ids] = .... 
    run_detector(test_scn_path, w, b, feature_params)
% 'test_scn_path' is a string. This directory contains images which may or
%    may not have faces in them. This function should work for the MIT+CMU
%    test set but also for any other images (e.g. class photos)
% 'w' and 'b' are the linear classifier parameters
% 'feature_params' is a struct, with fields
%   feature_params.template_size (probably 36), the number of pixels
%      spanned by each train / test template and
%   feature_params.hog_cell_size (default 6), the number of pixels in each
%      HoG cell. template size should be evenly divisible by hog_cell_size.
%      Smaller HoG cell sizes tend to work better, but they make things
%      slower because the feature dimensionality increases and more
%      importantly the step size of the classifier decreases at test time.

% 'bboxes' is Nx4. N is the number of detections. bboxes(i,:) is
%   [x_min, y_min, x_max, y_max] for detection i. 
%   Remember 'y' is dimension 1 in Matlab!
% 'confidences' is Nx1. confidences(i) is the real valued confidence of
%   detection i.
% 'image_ids' is an Nx1 cell array. image_ids{i} is the image file name
%   for detection i. (not the full path, just 'albert.jpg')

% The placeholder version of this code will return random bounding boxes in
% each test image. It will even do non-maximum suppression on the random
% bounding boxes to give you an example of how to call the function.

% Your actual code should convert each test image to HoG feature space with
% a _single_ call to vl_hog for each scale. Then step over the HoG cells,
% taking groups of cells that are the same size as your learned template,
% and classifying them. If the classification is above some confidence,
% keep the detection and then pass all the detections for an image to
% non-maximum suppression. For your initial debugging, you can operate only
% at a single scale and you can skip calling non-maximum suppression.

test_scenes = dir( fullfile( test_scn_path, '*.jpg' ) );
images      = readall( imageDatastore(test_scn_path) );
num_images  = length(images);

%initialize these as empty and incrementally expand them.
bboxes      = zeros(0,4);
confidences = zeros(0,1); 
image_ids   =  cell(0,1);

fpts  = feature_params.template_size; 
fphcs = feature_params.hog_cell_size;
window_size = fpts/fphcs;
tmplt_dim   = window_size^2 * 31;

for i = 1:num_images
    fprintf('\nDetecting faces in %s', test_scenes(i).name);
    
    img = images{i};
    [original_height, original_width] = size(img);

    fprintf('  (dims: %dx%d)\n', original_height, original_width);

    tic;
    
    cur_bboxes      = zeros(0,4);
    cur_confidences = zeros(0,1);
    cur_image_ids   =  cell(0,1);
    total_faces = 0;
    total_windows = 0;

    scale_factor = 10/11;
    max_scale = -log(min(size(img)))/log(scale_factor);
    scales = scale_factor.^(0:max_scale);
    
    for scale = scales
        
        scaled_img = im2single( imresize(img, scale) );
        
        [scaled_height, scaled_width] = size(scaled_img);
        
        hog = vl_hog(scaled_img, fphcs);
        [height,width,~] = size(hog);
        hog = reshape(hog, height*width, 31);
    
        height_end  = height - window_size;
        width_end   =  width - window_size;
        num_windows = (height_end+1)*(width_end+1);

        if (height_end < 0) || (width_end < 0)
            break;
        end
    
        total_windows = total_windows + num_windows;

        window_template = reshape( (0:window_size-1)*height + (1:window_size)', window_size^2, 1 );
        shift           = reshape( (0:width_end)*height + (0:height_end)', 1, num_windows );
        window_indices  = window_template + shift;
        
        hog_windows = hog(window_indices,:);
    
        hog_windows = permute( ...
            reshape( ...
                hog_windows, ...
                window_size^2, ...
                num_windows, ...
                31 ...
            ), ...
            [2 1 3] ...
        );
    
        hog_windows = reshape(hog_windows, num_windows, tmplt_dim);
        
        scores = hog_windows*w + b;
        
        windows_with_faces = find( scores>0.5 );                                 

        if isempty(windows_with_faces)
            continue;
        end
    
        x = mod( windows_with_faces-1, height_end+1 );
        y = floor( (windows_with_faces-1)/(height_end+1) );

        xmin = original_height *    (1 + fphcs*x) / scaled_height;
        xmax = original_height * (fpts + fphcs*x) / scaled_height;
        ymin = original_width  *    (1 + fphcs*y) / scaled_width;
        ymax = original_width  * (fpts + fphcs*y) / scaled_width;

        cur_bboxes      = [cur_bboxes; [ymin, xmin, ymax, xmax]];
        cur_confidences = [cur_confidences; scores(windows_with_faces)];
        cur_image_ids( total_faces + (1:length(windows_with_faces)), 1 ) = {test_scenes(i).name};
        total_faces = total_faces + length(windows_with_faces);
    end

    %non_max_supr_bbox can actually get somewhat slow with thousands of
    %initial detections. You could pre-filter the detections by confidence,
    %e.g. a detection with confidence -1.1 will probably never be
    %meaningful. You probably _don't_ want to threshold at 0.0, though. You
    %can get higher recall with a lower threshold. You don't need to modify
    %anything in non_max_supr_bbox, but you can.
    [is_maximum] = non_max_supr_bbox(cur_bboxes, cur_confidences, size(img));

    cur_confidences = cur_confidences(is_maximum,:);
    cur_bboxes      = cur_bboxes(     is_maximum,:);
    cur_image_ids   = cur_image_ids(  is_maximum,:);
    
    t = toc;
    fprintf('After examining %d windows, %d faces were detected in %f seconds\n', total_windows, length(cur_bboxes), t);


    bboxes      = [bboxes;      cur_bboxes];
    confidences = [confidences; cur_confidences];
    image_ids   = [image_ids;   cur_image_ids];

end




