function coded = codeit(txt)
    coded = arrayfun(@(x) change(x), txt);
end

function changed = change(x)
    changed = x;
    if x >= 'a' & x <= 'z'
        changed = char('z' + 'a' - x);
    elseif x >= 'A' & x <= 'Z'
        changed = char('Z' + 'A' - x);
    end
end