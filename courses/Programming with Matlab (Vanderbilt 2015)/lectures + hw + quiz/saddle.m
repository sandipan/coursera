function indices = saddle(M)
    indices = [];
    rows = [];
    cols = [];
    for j = 1:size(M, 2)
        for i = 1:size(M, 1)
            %if any(cols == j)
            %    continue
            %end
            if M(i, j) == max(M(i, :)) & M(i, j) == min(M(:, j)) 
                rows = [rows i];
                cols = [cols j];
                %break
            end
        end
    end
    for i = 1:numel(rows)
        indices = [indices; rows(i) cols(i)];
    end
end

function indices = saddle2(M)
    indices = [];
    rows = 1:size(M,1);
    cols = 1:size(M,2);
    for c = 1:size(M,2)
        [e, r] = min(M(rows, c))
        if M(r, c) == max(M(r, cols))
            indices = [indices; r c];
            rows = rows(rows~=r); %M(r, :) = [];
            cols = cols(cols~=c);
        end
        rows
    end
end

function indices = saddle1(M)
    indices = [];
    [e,c] = max(M, [], 2);
    for r = 1:size(M,1)
        if M(r, c(r)) == min(M(:, c(r)))
            indices = [indices; r c(r)];
        end
    end
end
