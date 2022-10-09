cd 'C:\courses\Coursera\Current\Image Processing\Week3'

X_k1k2 = 0;
N1 = 3;
N2 = 3;
k1 = 1;
k2 = 2;
for n1=0:(N1-1)
    for n2=0:(N2-1)
        X_k1k2 = X_k1k2 + (-1)^(n1+n2)*exp(-(2*pi/N1)*n1*k1*i)*exp(-(2*pi/N2)*n2*k2*i);
    end
end
X_k1k2

x = [1:4;5:8;9:12;13:16]';
X_k1k2 = 0;
N1 = 4;
N2 = 4;
k1 = 0;
k2 = 0;
for n1=0:(N1-1)
    for n2=0:(N2-1)
        X_k1k2 = X_k1k2 + x(n1+1,n2+1)*exp(-(2*pi/N1)*n1*k1*i)*exp(-(2*pi/N2)*n2*k2*i);
    end
end
X_k1k2

x = [2 1 1; 1 0 0; 0 0 0; 1 0 0];
X = zeros(4, 3);
N1 = 4;
N2 = 3;
for k1=0:(N1-1)
    for k2=0:(N2-1)
        for n1=0:(N1-1)
            for n2=0:(N2-1)
                X(k1+1,k2+1) = X(k1+1,k2+1) + x(n1+1,n2+1)*exp(-(2*pi/N1)*n1*k1*i)*exp(-(2*pi/N2)*n2*k2*i);
            end
        end
    end
end
X


I = imread('digital-images-week3_quizzes-original_quiz.jpg');

I2 = double(I); 
lpf = repmat(1/9, 3, 3);
I3 = imfilter(I2, lpf, 'replicate');
I4 = I3(1:2:end,1:2:end);
I5 = uint8(I4);
imwrite(I5, 'downsampled.jpg');

I6 = zeros(359,479);
%for i=1:2:479 % step size 2 - select odd indices
%    for j=1:2:359
for i=1:359
    for j=1:479
        if mod(i, 2) == 1 && mod(j, 2) == 1
            I6(i, j) = I4((i+1)/2, (j+1)/2);
        end
    end
end
ipf = [0.25,0.5,0.25;0.5,1,0.5;0.25,0.5,0.25];
I7 = imfilter(I6, ipf);
I8 = uint8(I7); %round(I7, 2);
imwrite(I8, 'upsampled.jpg');

N1 = size(I2,1);
N2 = size(I2,2);
MSE = 0;
%I9 = double(I8);
for i=1:N1
    for j=1:N2
        MSE = MSE + (I2(i, j) - I7(i, j))^2;
    end
end
MSE = MSE / (N1 * N2);
PSNR = 10*log10(255^2/MSE);
