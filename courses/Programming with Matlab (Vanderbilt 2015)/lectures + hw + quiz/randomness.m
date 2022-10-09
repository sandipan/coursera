function M = randomness(limit, n, m)
    M = floor(rand(n, m) .* limit + 1);  
    %hist(M)
end