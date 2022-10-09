function w = replace_me(v,a,b,c)
    if ~exist('b','var')
      b = 0;
    end
    if ~exist('c','var')
      c = b;
    end
    i = 1;
    j = 1;
    w = v;
    while i <= size(v, 2)
        if v(i) == a
            w(j) = b;
            w = [w(1:j) c w(j + 1:end)];
            j = j + 1;
        end
        i = i + 1;
        j = j + 1;
    end
end