function nothreaten = queen_check(board)
    [qposrow, qposcol] = ind2sub(size(board), find(board));
    nqueen = numel(qposrow);
    nothreaten = true;
    for i = 1:nqueen
        for j = i+1:nqueen
            qirow = qposrow(i);
            qicol = qposcol(i);
            qjrow = qposrow(j);
            qjcol = qposcol(j);
            if (qirow - qjrow == 0) | (qicol - qjcol == 0) | ...
               (abs(qirow - qjrow) == abs(qicol - qjcol))
               nothreaten = false;
               return;
            end
        end
    end
end
