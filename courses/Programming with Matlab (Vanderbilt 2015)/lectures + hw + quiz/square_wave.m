function sums = square_wave(n)
   sums = [];
   t = 0;
   for i = 1:1001
       sum = 0; 
       for k = 1:n
           sum = sum + sin((2 * k - 1) * t) / (2 * k - 1);
       end
       sums = [sums sum]; 
       t = t + 4 * pi / 1000;
   end
end

%x = square_wave(200);
%plot(1:1001, x);