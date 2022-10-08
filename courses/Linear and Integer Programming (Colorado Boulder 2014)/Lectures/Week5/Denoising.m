clear all
clc
close all

%% Make Corrupted Data
n = 4000;
t = (1:n)'; %matrix(list(range(n)), tc='d')
ex = 0.5 * sin(2*pi/n * t).*sin(0.01 * t);    %0.5 * mul( sin(2*pi/n * t), sin(0.01 * t))
corr = ex + 0.05 * randn(n,1);%ex + 0.05 * normal(n,1)

plot(t,corr)


%% Make the D matrix
D = sparse(n-1,n);
D(:,1:n-1) = -speye(n-1);
D(:,2:n) = D(:,2:n) + speye(n-1);

mu = 100;

%% Build the A = [I; sqrt(mu)D] matrix
A = [speye(n);
     sqrt(mu)*D];
 
b = [corr;
     zeros(n-1,1)];
 
%% Solve the Least-Squares Problem 
Sol = A\b; % or pinv(A)*b

hold on
plot(t,Sol,'r','linewidth',2)

set(gca,'fontsize',18)