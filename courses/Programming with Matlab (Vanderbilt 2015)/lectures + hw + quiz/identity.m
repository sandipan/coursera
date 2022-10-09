function I = identity(n)
    I = zeros(n);
    for i=1:n
        I((i-1) * n + i) = 1;
    end
end