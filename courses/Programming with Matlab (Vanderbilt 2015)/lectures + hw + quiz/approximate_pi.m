function [api, k] = approximate_pi(delta)
   api = 0;
   powm3 = 1;
   i = 0;
   while 1
       api = api + sqrt(12) * powm3 / (2 * i + 1);
       powm3 = powm3 * (-1 / 3);
       if abs(pi - api) <= delta
           k = i;
           break
       end
       i = i + 1;
   end   
end