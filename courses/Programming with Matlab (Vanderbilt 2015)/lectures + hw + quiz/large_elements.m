function indexes = large_elements(A)
    m = size(A, 1);
    n = size(A, 2);
    B = reshape(A', 1, n * m);
    N = size(B, 2);
    indexes = [];
    for i = 1:N
        row = ceil(i / n);
        col = mod(i - 1, n) + 1;
        if B(i) > row + col  
            indexes = [indexes; row col];
        end
    end
end