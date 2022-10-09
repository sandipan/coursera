function count = letter_counter(filename)
    count = -1;
    fileID = fopen(filename);
    if (fileID ~= -1)
       chars = fscanf(fileID, '%c');
       count = sum(isletter(chars));
    end
end
