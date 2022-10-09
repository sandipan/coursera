cd 'C:\courses\Coursera\Current\Image Processing\Week8'

% entropy([0.2 0.5 0.3])
% 0.25 + 0.8*(0.4-0.25)
% (0.37 + 0.4) / 2
% 0.37 + 0.5*(0.4-0.37)
% 0.37 + 0.8*(0.4-0.37)

I = imread('Cameraman256.bmp');
imshow(double(I) / 255);

m = size(I, 1);
n = size(I, 2);

probs = zeros(1, 256);
for i = 1:m
    for j = 1:n
        probs(I(i,j)+1) = probs(I(i,j)+1) + 1;
    end
end
probs = probs / (m*n);
sum(probs)
entropy(probs)