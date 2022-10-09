cd 'C:\courses\Coursera\Current\Image Processing\Week4'

B1 = [1 1 2 2; 1 1 2 2; 2 2 3 4; 2 2 5 6];
B2 = [2 2 1 1; 2 2 2 2; 2 2 6 4; 2 2 5 3];
mean2((B1 - B2).*(B1 - B2))

m = size(B1, 1);
n = size(B1, 2);
MSE = 0;
for i=1:m
    for j=1:n
        MSE = MSE + (B1(i,j) - B2(i,j))^2;
    end
end
MSE / (m * n)

x = [10 20 10 10; 20 40 10 10; 30 40 20 20; 50 60 20 20];
y1 = [10 20 10 10; 20 40 10 10; 20 20 30 40; 20 20 50 60];
y2 = [20 30 20 20; 30 50 20 20; 40 50 30 30; 60 70 30 30];
y3 = [10 20 30 40; 20 40 50 60; 10 10 20 20; 10 10 20 20];
y4 = [1 2 1 1; 2 4 1 1; 3 4 2 2; 5 6 2 2];
mean2(abs(x - y1))
mean2(abs(x - y2))
mean2(abs(x - y3))
mean2(abs(x - y4))

I = imread('digital-images-week4_quizzes-frame_1.jpg');
I1 = double(I); 
I = imread('digital-images-week4_quizzes-frame_2.jpg');
I2 = double(I); 
B_target = I2(65:96,81:112);
M = size(I1, 1);
N = size(I1, 2);
min_mae = Inf;
min_block = [-1 -1];
for i=1:(M-31)
    for j=1:(N-31)
        B1 = I1(i:(i+31),j:(j+31));
        mae = mean2(abs(B_target - B1));
        if mae < min_mae
            min_mae = mae;
            min_block = [i j];
        end
    end
end

min_mae
min_block

for i=65:96
    I2(i, 81) = 0;
    I2(i, 112) = 0;
end
for j=81:112
    I2(65, j) = 0;
    I2(96, j) = 0;
end
I2 = uint8(I2);
imwrite(I2, 'digital-images-week4_quizzes-frame_2_with_block.jpg');

block_i = min_block(1);
block_j = min_block(2);
for i=block_i:(block_i+31)
    I1(i, block_j) = 0;
    I1(i, block_j+31) = 0;
end
for j=block_j:(block_j+31)
    I1(block_i, j) = 0;
    I1(block_i+31, j) = 0;
end
I1 = uint8(I1);
imwrite(I1, 'digital-images-week4_quizzes-frame_1_with_block.jpg');
