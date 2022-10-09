function hindu = roman(str)
    hindu = uint8(0);
    numMap = containers.Map(...
    {'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X',...
    'XI', 'XII', 'XIII', 'XIV', 'XV', 'XVI', 'XVII', 'XVIII', 'XIX', 'XX'},...
    num2cell(1:20));
    if isKey(numMap, str)
        hindu = uint8(numMap(str));
    end
end
