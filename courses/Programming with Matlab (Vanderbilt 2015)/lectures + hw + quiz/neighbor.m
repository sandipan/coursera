function nv = neighbor(v)
    nv = [];
    m = size(v, 1);
    n = size(v, 2);
    if m == 1 & n >= 2
        for i = 2:n
            nv(end + 1) = abs(v(i - 1) - v(i));
        end
    end
end