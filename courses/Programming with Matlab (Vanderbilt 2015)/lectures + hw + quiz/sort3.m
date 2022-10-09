function v = sort3(a, b, c)
    v = [a b c];
    if a < b
        if c < a
            v = [c a b];
        elseif c < b
            v = [a c b];
        end
    else
        if b > c
            v = [c b a];
        elseif a > c
            v = [b c a];
        else
            v = [b a c];
        end
    end
end