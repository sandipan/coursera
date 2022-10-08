% This code is based on the code found on the CVX and CVXOPT example pages
clear all
clc
close all

randn('state',0);
m=200; n=80;
A = randn(m,n);
b = randn(m,1);

% ell_1 approximation
% minimize   ||Ax-b||_1
%
% cvx_begin
%     variable x1(n)
%     minimize(norm(A*x1-b,1))
% cvx_end
c_Big = [zeros(n,1);
         ones(m,1)];
A_Big = [A -eye(m);
        -A -eye(m)];
b_Big = [b;
        -b];
Hold = linprog(c_Big,A_Big,b_Big);
x1 = Hold(1:n);

% ell_2 approximation
% minimize ||Ax-b||_2
x2=A\b;

% ell_inf approximation
% minimize   ||Ax-b||_inf
%
% cvx_begin
%     variable x3(n)
%     minimize(norm(A*x3+b,inf))
% cvx_end
c_Big3 = [zeros(n,1);
          1];
A_Big3 = [A -ones(m,1);
        -A -ones(m,1)];
b_Big3 = [b;
        -b];
Hold3 = linprog(c_Big3,A_Big3,b_Big3);
x3 = Hold3(1:n);


% Plot histogram of residuals

ss = max(abs([A*x1+b; A*x2+b; A*x3+b]));
tt = -ceil(ss):0.05:ceil(ss);  % sets center for each bin
[N1,hist1] = hist(A*x1-b,tt);
[N2,hist2] = hist(A*x2-b,tt);
[N3,hist3] = hist(A*x3-b,tt);


range_max=2.0;  
rr=-range_max:1e-2:range_max; %% rr is For Drawing the Function

figure(1), clf, hold off
subplot(3,1,1),
bar(hist1,N1);
hold on
plot(rr, abs(rr)*40/3, '-'); %plots the absolute value cost function
ylabel('l_1','FontSize',18)
axis([-range_max range_max 0 40]);
hold off

subplot(3,1,2),
bar(hist2,N2);
hold on;
plot(rr,2*rr.^2), %plots the quadratic cost function
ylabel('l_2','FontSize',18)
axis([-range_max range_max 0 11]);
hold off

subplot(3,1,3),
bar(hist3,N3);
ylabel('l_\infty','FontSize',18)
axis([-range_max range_max 0 25]);
xlabel('residual','FontSize',18)

%set(gca,'FontSize',18)
