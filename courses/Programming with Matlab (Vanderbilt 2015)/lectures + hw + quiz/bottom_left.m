function S = bottom_left(N, n)
    S = N(size(N, 1) - n + 1:size(N, 1), 1:n);
end