cd 'C:\courses\Coursera\Current\Image Processing\Week2'

x = [4 5 6; 1 2 3];
h = [1 1; 1 1];
y = conv2(x, h);
y

x = [0 0 6; 0 4 5; 1 2 3];
h = [1 1; 1 1];
y = conv2(x, h);
y

I = imread('digital-images-week2_quizzes-lena.gif');

I2 = double(I); %im2uint8(I);
lpf = repmat(1/9, 3, 3);
I3 = imfilter(I2, lpf, 'replicate');
imwrite(I3, 'lena2.gif');

N1 = size(I2,1);
N2 = size(I2,2);
MSE = 0;
for i=1:N1
    for j=1:N2
        MSE = MSE + (I3(i, j) - I2(i, j))^2;
    end
end
MSE = MSE / (N1 * N2);
PSNR = 10*log10(255^2/MSE);

lpf2 = repmat(1/25, 5, 5);
I3 = imfilter(I2, lpf2, 'replicate');
imwrite(I3, 'lena3.gif');

N1 = size(I2,1);
N2 = size(I2,2);
MSE = 0;
for i=1:N1
    for j=1:N2
        MSE = MSE + (I3(i, j) - I2(i, j))^2;
    end
end
MSE = MSE / (N1 * N2);
PSNR = round(10*log10(255^2/MSE), 2);