function [even, odd] = separate_by_two(A)
   even = A(mod(A, 2) == 0)';
   odd = A(mod(A, 2) == 1)';
end