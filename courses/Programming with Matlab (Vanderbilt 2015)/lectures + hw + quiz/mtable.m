function [mt, s] = mtable(n, m)
    s = 0;
    mt = zeros(n, m);
    for i=1:n
        for j=1:m
            mt(i, j) = i * j;
            s = s + mt(i, j);
        end
    end
end