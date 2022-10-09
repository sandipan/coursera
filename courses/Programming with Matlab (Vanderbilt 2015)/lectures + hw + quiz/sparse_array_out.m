function opened = sparse_array_out(arr, filename)
    fileID = fopen(filename, 'w');
    opened = (fileID ~= -1);
    if opened
        [row, col] = ind2sub(size(arr), find(arr));
        fwrite(fileID, size(arr, 1),'uint32');
        fwrite(fileID, size(arr, 2),'uint32');
        fwrite(fileID, numel(row),'uint32');
        for i=1:numel(row)
         fwrite(fileID, row(i),'uint32');
         fwrite(fileID, col(i),'uint32');
         fwrite(fileID, arr(row(i), col(i)),'double');
        end
        fclose(fileID);
    end
end
