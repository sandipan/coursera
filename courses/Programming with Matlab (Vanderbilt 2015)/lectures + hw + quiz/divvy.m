function B = divvy(A, k)
   B = A;
   B(mod(B, k) ~= 0) = B(mod(B, k) ~= 0) * k;
end