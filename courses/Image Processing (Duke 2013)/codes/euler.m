clear;
%  Coefficients of equation a x'=b x + c t
a=1.;
b=-2.;
c=1.;

%  Initial and Final Times
tinit= 0.;
tmax=5.; 

%  Number of Time Steps
maxt = 3000;
dt = (tmax-tinit)/maxt;

%  Initial Condition
x(1)=1.;
t(1)=tinit;

%  Time Loop
for j=1:maxt;
x(j+1)=x(j)+dt*((b*x(j)+c*t(j))/a);
t(j+1)=tinit+j*dt;
end;

plot(t,x)
title('Euler Method')
xlabel('T')
ylabel('X(t)')