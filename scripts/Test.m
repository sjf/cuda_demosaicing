%
% Test the alternating projections algorithm
%

clear all;

% Read image
a = imread('16small.tif');

% Test the algorithm
% Number of iterations and the filters used can be changed within the demopocs code..

[out_pocs,out_bilinear] = demopocs(a);

% Display images
figure; imshow(uint8(a)); title('Original');
figure; imshow(uint8(out_pocs)); title('Demosaicing using alternating projections');
figure; imshow(uint8(out_bilinear)); title('Bilinear interpolation');


