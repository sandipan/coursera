% Purpose: Reconstructs an image from its horizontal and vertical gradient,
%     plus one pixel value.
% 
% Author: Aaron Smith
% Class: COMP 790 Computational Photography, FA14
% Project: Assignment 2
% Due Date: 18 Sep 2014

% Clear Workspace
clear all

% Image Filename
fp = 'toygc.png';

% Read Image Into Memory
I = im2double(rgb2gray(imread(fp)));

% Get Image Dimensions
[h, w] = size(I);

% Calculate Number of Equations To Produce
ne = 2*h*w - h - w + 1;

% Create Sparse Objective Matrix A
A = spalloc(ne, h*w, 2*ne);

% Create Result Vector b
b = zeros(ne, 1);

% Initialize Equation Counter
e = 0;

% Create Matrix For Easy Subscript-To-Linear-Index Conversion
im2var = reshape(1:h*w, [h, w]);

% Minimize Horizontal Gradient Error
for x = 1:w-1
    for y = 1:h
        e = e + 1;
        A(e, im2var(y, x+1)) = 1;
        A(e, im2var(y, x)) = -1;
        b(e) = I(im2var(y, x+1)) - I(im2var(y, x));
    end
end

% Minimize Vertical Gradient Error
for x = 1:w
    for y = 1:h-1
        e = e + 1;
        A(e, im2var(y+1, x)) = 1;
        A(e, im2var(y, x)) = -1;
        b(e) = I(im2var(y+1, x)) - I(im2var(y, x));
    end
end

% Explicitly Specify One Pixel
e = e + 1;
A(e, 1) = 1;
b(e) = I(1);

% Solve System
v = reshape(lscov(A, b), [h, w]);

% Show Resulting Image
imshow(v);