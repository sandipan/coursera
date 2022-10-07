% Purpose: Blends two images with a mask, without employing smoothing
%     techniques.
% 
% Author: Aaron Smith
% Class: COMP 790 Computational Photography, FA14
% Project: Assignment 2
% Due Date: 18 Sep 2014

% Clear Workspace
clear all

% Image Filenames
fpSource = 'source.png';
fpTarget = 'target.png';
fpMask = 'mask.png';
%fpSource = 'source.jpg';
%fpTarget = 'target.jpg';
%fpMask = 'mask.jpg';


% Read Images Into Memory
S = im2double(imread(fpSource));
T = im2double(imread(fpTarget));
M = im2double(imread(fpMask));

% Check Image Dimensions
if ~isequal(size(S), size(T))
    error('Error: Source and target images are not the same size.')
elseif ~isequal(size(S), size(M))
    error('Error: Mask image is not the same size as source and target images')
end

% Apply Image Cloning
I = M.*S + (1-M).*T;

% Show Result
figure
imshow(I);