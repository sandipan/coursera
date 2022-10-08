clear all
clc
close all

% Make noising data
m = 4;
b = -10;
n = 50;
v = 10;

Rand = v*rand(1,n)';
x = linspace(1,10,n)';
y = m*x + b + Rand;

% Add outliers to the data
y2=y;
y2(1)=60;
y2(50)=-60;

% Plot Data with outliers
plot(x,y2,'.')

A = [x ones(n,1)];
u1 = A\y; % Least-Squares Sol without outliers
u2 = A\y2; % Least-Squares Sol with outliers

% l_inf with outliers
cvx_begin
    variables r(2);
    minimize( norm(A*r-y2,inf));
cvx_end

% l_1 with outliers
cvx_begin
    variables s(2);
    minimize( norm(A*s-y2,1));
cvx_end

% Plot the lines defined by the solutions
M1 = u1(1);
M2 = u2(1);
M3 = r(1);
M4 = s(1);
B1 = u1(2);
B2 = u2(2);
B3 = r(2);
B4 = s(2);

hold on
plot(x,M1.*x+B1,'c')
plot(x,M2.*x+B2,'r--')
plot(x,M3.*x+B3,'g')
plot(x,M4.*x+B4,'k--')
xlabel('X')
ylabel('Y')
title('Y = m*X + b')

set(gca,'FontSize',14)
set(get(gca,'XLabel'),'FontSize',14)
set(get(gca,'YLabel'),'FontSize',14)
set(get(gca,'Title'),'FontSize',14)
set(get(gca,'Children'),'LineWidth',2)
set(get(gca,'Children'),'MarkerSize',18)
legend('Data','Least-Squares (LS)','LS w/ Outliers','l_\infty','l_1','Location','Best')

