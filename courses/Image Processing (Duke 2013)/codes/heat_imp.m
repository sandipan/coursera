%  Heat Diffusion in one dimensional wire within the Fully Implicit Method
clear;

% Parameters to define the heat equation and the range in space and time
L = 1.;  		%  Lenth of the wire
T =1.;			%  Final time


%  Parameters needed to solve the equation within the fully implicit method
maxk = 2500;					%  Number of time steps
dt = T/maxk;
n = 50.;							%  Number of space steps
dx = L/n;
cond = 1/2;						%  Conductivity
b = cond*dt/(dx*dx);		 	%  Parameter of the method 

%  Initial temperature of the wire: a sinus.
for i = 1:n+1
        x(i) =(i-1)*dx;
        u(i,1) =sin(pi*x(i));        
end
          
%  Temperature at the boundary (T=0)
for k=1:maxk+1
        u(1,k) = 0.;
        u(n+1,k) = 0.;
        time(k) = (k-1)*dt;
end


aa(1:n-2)=-b;
bb(1:n-1)=1.+2.*b;
cc(1:n-2)=-b;
MM=diag(bb,0)+diag(aa,-1)+diag(cc,1);


%  Implementation of the implicit method
for k=2:maxk		%  Time Loop   
   uu=u(2:n,k-1);
   u(2:n,k)=inv(MM)*uu;
end

% Graphical representation of the temperature at different selected times
figure(1)
plot(x,u(:,1),'-',x,u(:,100),'-',x,u(:,300),'-',x,u(:,600),'-')
title('Temperature within the fully implicit method')
xlabel('X')
ylabel('T')

figure(2)
mesh(x,time,u')
title('Temperature within the fully implicit method')
xlabel('X')
ylabel('Temperature')
