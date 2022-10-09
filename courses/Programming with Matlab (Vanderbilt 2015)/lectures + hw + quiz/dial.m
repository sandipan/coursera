function coded = dial(number)
    numMap = containers.Map(...
    num2cell(setdiff([' ', '#', '(', ')', '*', '-', '0':'9', 'A':'Z'], ['Q','Z'])),...
    num2cell([' ', '#', ' ', ' ', '*', ' ', '0':'9', repmat('2',1,3),repmat('3',1,3),...
    repmat('4',1,3),repmat('5',1,3),repmat('6',1,3),repmat('7', 1, 3),...
    repmat('8',1,3),repmat('9',1,3)]));
    coded = number;
    for i=1:numel(number)
        if isKey(numMap, number(i))
            coded(i) = numMap(number(i));
        else
            coded = [];
            break;
        end
    end
    %coded = arrayfun(@(x) change(x, numMap), number);
end

function changed = change(x, numMap)
    changed = ' ';
    if isKey(numMap, x)
        changed = numMap(x);
    end
end