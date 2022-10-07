% Solving the a x'=b x + c t ODE by using the midpoint method
clear;

%  Coefficients of equation a x'=b x + c t
a=1.;
b=-2.;
c=1.;

%  Initial and Final Times
tinit= 0.;
tmax=5.; 

%  Number of Time Steps
maxt = 30000;
dt = (tmax-tinit)/maxt;

%  Initial Condition
x(2)=1.;
x(1)=1.0-dt*((b*x(2)+c*tinit)/a);
t(2)=tinit;

%  Time Loop
for j=2:(maxt+1);
x(j+1)=x(j-1)+2.*dt*((b*x(j)+c*t(j))/a);
t(j+1)=tinit+(j-1)*dt;
end;

plot(t,x)
title('Midpoint Method')
xlabel('T')
ylabel('X(t)')