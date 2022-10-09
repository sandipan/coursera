cd 'C:\courses\Coursera\Current\Image Processing\Week12'
I = eye(10);
D = zeros(10);
for i = 1:10
    for j = 1:10
        D(i, j) = sin(i + j);
    end
end
A = D + I;
b = [-2,-6,-9,1,8,10,1,-9,-4,-3]';
S = 3;
indices = 1:10;
r = b;
xs = zeros(1,size(A,2));
%A = A / normc(A);
for i = 1:size(A,2)
    A(:,i) = A(:,i) / sqrt(sum(A(:,i).^2));
end

while S > 0
    maxP = -1;
    imax = -1;
    for i = 1:size(indices,2)
        xstar = A(:,indices(i))' * r; % / (A(:,i)'*A(:,i));
        if xstar > maxP
            maxP = xstar;
            imax = indices(i);
        end
    end
    imax
    %x = A(:,imax) \  r;
    %x
    %maxP
    %r = r - x * A(:,imax);
    r = r - maxP * A(:,imax);
    xs(imax) = maxP;
    S = S - 1;
    indices = setdiff(indices, imax);
end
xs