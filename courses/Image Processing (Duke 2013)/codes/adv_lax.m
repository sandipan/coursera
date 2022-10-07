%  Square-wave Test for the Lax method to solve the Advection Equation
clear;

% Parameters to define the advection equation and the range in space and time
Lmax = 1.0; 	% Maximum length
Tmax = 1.;    	% Maximum time
c = 1.0;    	% Advection velocity

%  Parameters needed to solve the equation within the Lax method
maxt = 300; 		% Number of time steps
dt = Tmax/maxt;		
n = 300;				%  Number of space steps
nint=50;				%  The wave-front: intermediate point from which u=0 (nint<n)!!
dx = Lmax/n;			
b = c*dt/(2.*dx);
% The Lax method is stable for abs(b)=< 1/2 but it gets difussed unless abs(b)= 1/2


%  Initial value of the function u (amplitude of the wave)

for i = 1:(n+1)
   if i < nint
      u(i,1)=1.;
   else
      u(i,1)=0.;
   end 
   x(i) =(i-1)*dx;
end



%  Value of the amplitude at the boundary at any time
for k=1:maxt+1
        u(1,k) = 1.;
        u(n+1,k) = 0.;
        time(k) = (k-1)*dt;
end


%  Implementation of the Lax method
for k=1:maxt		%  Time loop
   for i=2:n		%  Space loop
      u(i,k+1) =0.5*(u(i+1,k)+u(i-1,k))-b*(u(i+1,k)-u(i-1,k));
   end
end

% Graphical representations of the evolution of the wave
figure(1)     
mesh(x,time,u')
title('Square-wave test within the Lax Method')
xlabel('X')
ylabel('T')

figure(2)
plot(x,u(:,1),'-',x,u(:,20),'-',x,u(:,50),'-',x,u(:,100),'-')
title('Square-wave test within the Lax Method')
xlabel('X')
ylabel('Amplitude(X)')
