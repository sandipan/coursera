function n = one_per_n(x)
   sum = 0;
   for i = 1:10000
       sum = sum + 1 / i;
       if sum >= x
           n = i;
           break
       end
   end   
   if sum < x
       n = -1;
   end    
end