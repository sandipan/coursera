function M = bell(n)
    if n <= 0 | n - ceil(n) ~= 0 %~isinteger(uint8(n)) %
        M = [];
    else    
        M = zeros(n);
        M(1, 1) = 1;
        i_new = 2;
        while i_new <= n
            M(i_new, 1) = M(1, i_new - 1);
            i = i_new - 1;
            j = 2;
            while i >= 1
                M(i, j) = M(i, j - 1) + M(i + 1, j - 1);
                i = i - 1;
                j = j + 1;
            end
            i_new = i_new + 1;
        end
    end
end
