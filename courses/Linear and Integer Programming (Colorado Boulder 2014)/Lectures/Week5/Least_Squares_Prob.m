clear all
clc
close all

%% Make Corrupted Data
t = (0:.001:4)'; 
n = length(t);
y = 3 + 7.*t + 4.*t.^2 - t.^3;  
y_corr = y + 1*randn(n,1);

plot(t,y_corr)

%% Setup Ax = b, assuming that y = a0 + a1*t + a2*t^2 
%%                                  + a3*t^3 + a4*t^4 + a5*t^5
%% the variable x = [a0; a1; a2; a3; a4];

A = [ones(n,1) t t.^2 t.^3 t.^4];
b = y_corr;


%% Solve the Least-Squares Problem 
Sol = A\b; % or pinv(A)*b
y_Sol = Sol(1) + Sol(2).*t + Sol(3).*t.^2 + Sol(4)*t.^3 + Sol(5)*t.^4;  

hold on
plot(t,y_Sol,'r','linewidth',2)

set(gca,'fontsize',18)