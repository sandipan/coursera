function [rowa,cola,numrowsa,numcolsa,summa] = maxsubsum(A)
   m = size(A, 1);
   n = size(A, 2); 
   summa = -Inf;
   for row = 1:m
       for col = 1:n
           for numrows = 1 : m - row + 1
               for numcols = 1 : n - col + 1
                    csum = sum(sum(A(row : row + numrows - 1, col : col + numcols - 1)));
                    if csum > summa
                        rowa = row;
                        cola = col;
                        numrowsa = numrows;
                        numcolsa = numcols;
                        summa = csum;
                    end
               end
           end
       end
   end
end
