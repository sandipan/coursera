function S = halfsum(A)
    S = 0;
    for i = 1:size(A,1)
        for j = 1:size(A,2)
            if i <= j
                S = S + A(i, j);
            end    
        end
    end
end