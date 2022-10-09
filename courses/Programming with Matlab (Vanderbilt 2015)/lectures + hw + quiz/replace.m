function coded = replace(txts, c1, c2)
    coded = txts;
    for i = 1:numel(txts)
        coded{i} = arrayfun(@(x) change(x, c1, c2), txts{i});
    end
end

function changed = change(x, c1, c2)
    changed = x;
    if x == c1
        changed = c2;
    end
end