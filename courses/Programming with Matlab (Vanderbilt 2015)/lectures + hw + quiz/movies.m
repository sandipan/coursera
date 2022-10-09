function b = movies(hr1, min1, durmin1, hr2, min2, durmin2)
    start1 = hr1 * 60 + min1
    end1 = start1 + durmin1
    start2 = hr2 * 60 + min2
    end2 = start2 + durmin2
    b = 0
    if start1 <= start2
        b = (end1 <= start2) & (start2 - end1 <= 30);
    %else
    %    b = (end2 <= start1) & (start1 - end2 <= 30);
    end    
end