function gen = generationXYZ(year)
    if year < 1966
        gen = 'O';
    elseif year > 2012
        gen = 'K';
    elseif year >= 1966 & year <= 1980
        gen = 'X';
    elseif year >= 1981 & year <= 1999
        gen = 'Y';
    else 
        gen = 'Z';
    end
end