function v = classify(x)
    d1 = size(x, 1);
    d2 = size(x, 2);
    if d1 == 0 | d2 == 0
        v = -1;
    elseif d1 == 1 & d2 == 1
        v = 0;
    elseif (d1 > 1 & d2 == 1) | (d1 == 1 & d2 > 1)
        v = 1;
    else
        v = 2;
    end    
end