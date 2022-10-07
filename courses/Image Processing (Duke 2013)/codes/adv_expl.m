%  Square-wave Test for the explicit method to solve the Advection Equation
clear;

% Parameters to define the advection equation and the range in space and time
Lmax = 1.0; 	% Maximum length
Tmax = 1.;    	% Maximum time
c = 1.0;    	% Advection velocity

%  Parameters needed to solve the equation within the explicit method
maxt = 3000; 		% Number of time steps
dt = Tmax/maxt;		
n = 30;				%  Number of space steps
nint=15;				%  The wave-front: intermediate point from which u=0
dx = Lmax/n;			
b = c*dt/(2.*dx); 

%  Initial value of the function u (amplitude of the wave)
for i = 1:(n+1)
   if i < nint
      u(i,1)=1.;
   else
      u(i,1)=0.;
   end 
   x(i) =(i-1)*dx;
end

%  Value of the amplitude at the boundary 
for k=1:maxt+1
        u(1,k) = 1.;
        u(n+1,k) = 0.;
        time(k) = (k-1)*dt;
end


%  Implementation of the explicit method
for k=1:maxt		%  Time loop
   for i=2:n		%  Space loop
      u(i,k+1) =u(i,k)-b*(u(i+1,k)-u(i-1,k));
   end
end

% Graphical representation of the wave at different selected times
plot(x,u(:,1),'-',x,u(:,50),'-',x,u(:,200),'-',x,u(:,500),'-')
title('Square-wave Test within the Explicit Method I')
xlabel('X')
ylabel('Amplitude(X)')
