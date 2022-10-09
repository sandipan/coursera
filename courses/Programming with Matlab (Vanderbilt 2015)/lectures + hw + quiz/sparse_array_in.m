function arr = sparse_array_in(filename)
    fileID = fopen(filename);
    arr = [];
    if (fileID ~= -1)
        m = fread(fileID, 1, 'uint32');
        n = fread(fileID, 1, 'uint32');
        e = fread(fileID, 1, 'uint32');
        arr = zeros(m, n);
        for i=1:e
         row = fread(fileID, 1,'uint32');
         col = fread(fileID, 1,'uint32');
         val = fread(fileID, 1,'double');
         arr(row, col) = val;
        end
        fclose(fileID);
    end
end
