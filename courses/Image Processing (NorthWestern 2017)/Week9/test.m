cd 'C:\courses\Coursera\Current\Image Processing\Week9'

I = imread('Cameraman256.bmp');
imshow(double(I) / 255);
imwrite(I, 'Cameraman256.jpg', 'jpg', 'quality', 10);

I2 = imread('Cameraman256.jpg');
I2 = double(I2);
I = double(I);
N1 = size(I2,1);
N2 = size(I2,2);
MSE = 0;
for i=1:N1
    for j=1:N2
        MSE = MSE + (I2(i, j) - I(i, j))^2;
    end
end
MSE = MSE / (N1 * N2);
MSE = (norm(I2-I, 'fro'))^2 / (N1 * N2);
PSNR = 10*log10(255^2/MSE);