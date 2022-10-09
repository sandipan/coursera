function b = older(y1, m1, d1, y2, m2, d2)
    if y1 == y2 & m1 == m2 & d1 == d2
        b = 0;
    elseif (y1 > y2) | (y1 == y2 & m1 > m2) | (y1 == y2 & m1 == m2 & d1 > d2)
        b = -1;
    else
        b = 1;
    end    
end