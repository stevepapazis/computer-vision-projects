#! /usr/bin/octave -qf

pkg load image;
ignore_function_time_stamp("all");

addpath(fileparts(mfilename("fullpath")));

usage = "Usage: octave main.m <input> <output> <reduction direction {v|h}> <amount> <p for L_p or 0 for L_infinity> [--animations]\n";

args = argv();
if (length(args) == 5)
  create_animations = 0;
elseif (length(args) == 6 && strcmp(args{6}, "--animations"))
  create_animations = 1;
else
  printf(usage);
  quit();
endif

[input_path, output_path, direction, amount, energy_function] = args{1:5};

amount = str2double(amount);
energy_function = str2double(energy_function);

if (isnan(amount) || amount < 1 || isnan(energy_function) || energy_function < 0)
  printf(usage);
  quit();
endif
amount = uint64(amount);

im = imread(input_path);

if (direction == "h")
  [im, imgs, enMaps, minEnMaps] = reduceWidth(im, amount, energy_function, create_animations);
elseif (direction == "v")
  [im, imgs, enMaps, minEnMaps] = reduceHeight(im, amount, energy_function, create_animations);
else
  printf(usage);
  quit();
endif

[output_dir, file_name, ext] = fileparts(output_path);
imwrite(im, fullfile(output_dir, [file_name, ext]));
if (create_animations)
  saveAnimation(imgs, fullfile(output_dir, [file_name, ".mp4"]));
  saveAnimation(enMaps, fullfile(output_dir, [file_name, "_energy.mp4"]));
  saveAnimation(minEnMaps, fullfile(output_dir, [file_name, "_minEnergy.mp4"]));
endif