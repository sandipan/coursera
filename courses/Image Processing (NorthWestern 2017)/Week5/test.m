cd 'C:\courses\Coursera\Current\Image Processing\Week5'

% mean(X) = 120; sd(X) =  10;
% mean(255 - X) = 255 - 120 = 135; sd(255 - X) = 10;

% sum_{i=0}^{255}(i*p_x(i));
% sum_{i=0}^{255}(P_x(i)*p_x(i)) = sum_{i=0}^{255}(sum_{j=0}^{i}p_x(j)*p_x(i));

I = imread('digital-images-week5_quizzes-noisy.jpg');
I1 = double(I); 
imshow(I1 / 255);

I2 = medfilt2(I1);
imshow(I2 / 255);

I3 = medfilt2(I2);
imshow(I3 / 255);

I = imread('digital-images-week5_quizzes-original.jpg');
I4 = double(I); 
imshow(I4 / 255);

N1 = size(I1,1);
N2 = size(I1,2);
MSE = 0;
for i=1:N1
    for j=1:N2
        MSE = MSE + (I4(i, j) - I1(i, j))^2;
    end
end
MSE = MSE / (N1 * N2);
PSNR = 10*log10(255^2/MSE);

MSE = 0;
for i=1:N1
    for j=1:N2
        MSE = MSE + (I4(i, j) - I2(i, j))^2;
    end
end
MSE = MSE / (N1 * N2);
PSNR = 10*log10(255^2/MSE);

MSE = 0;
for i=1:N1
    for j=1:N2
        MSE = MSE + (I4(i, j) - I3(i, j))^2;
    end
end
MSE = MSE / (N1 * N2);
PSNR = 10*log10(255^2/MSE);
