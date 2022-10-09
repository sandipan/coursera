function censored = censor(cvec, str)
    censored = {};
    j = 1;
    for i = 1:numel(cvec)
        if numel(strfind(cvec{i}, str)) == 0
            censored{j} = cvec{i};
            j = j + 1;
        end
    end
end
