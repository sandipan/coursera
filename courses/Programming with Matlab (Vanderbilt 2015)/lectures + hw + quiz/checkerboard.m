function board = checkerboard(n, m)
    if mod(m, 2) == 1
        row1 = repmat([1 0], 1, (m - 1) / 2);
        row2 = repmat([0 1], 1, (m - 1) / 2);
        row1(m) = 1;
        row2(m) = 0;
    else
        row1 = repmat([1 0], 1, m / 2);
        row2 = repmat([0 1], 1, m / 2);
    end    
    if mod(n, 2) == 1
        board = [repmat([row1; row2], (n - 1) / 2, 1); row1];        
    else
        board = repmat([row1; row2], n / 2, 1);
    end    
end