function is_prime = myprime(n)
   is_prime = n > 1;
   for d = 2:floor(sqrt(n))
       if rem(n, d) == 0
           is_prime = false;
       break;    
   end
end

%x = square_wave(200);
%plot(1:1001, x);