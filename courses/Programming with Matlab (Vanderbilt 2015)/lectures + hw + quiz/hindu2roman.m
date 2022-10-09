function r = hindu2roman(n)
   n = str2num(n);
   r = '';
   for i = 1:floor(n / 100)
       r = [r 'C'];
   end
   n = rem(n, 100);
   if n >= 90
       r = [r 'XC'];
       n = n - 90;
   elseif n >= 50
       r = [r 'L'];
       n = n - 50;
   end
   if n >= 40
       r = [r 'XL'];
       n = n - 40;
   else    
       for i = 1:floor(n / 10)
           r = [r 'X'];
       end
       n = rem(n, 10); 
   end
   if n == 9
       r = [r 'IX'];
       n = n - 9;
   elseif n >= 5
       r = [r 'V'];
       n = n - 5;
   end
   if n == 4
       r = [r 'IV'];
   else
       for i = 1:n
           r = [r 'I'];
       end           
   end
end
