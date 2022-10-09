function avg = moving_average(x)
    persistent buffer;
    if size(buffer, 2) == 25
        buffer = buffer(2:end);
    end
    buffer(end + 1) = x;
    %size(buffer)
    avg = mean(buffer);
end