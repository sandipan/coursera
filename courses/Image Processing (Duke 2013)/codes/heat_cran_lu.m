%  Heat Diffusion in one dimensional wire within the Crank-Nicholson Method
%  by using the LU decomposition.
clear;

% Parameters to define the heat equation and the range in space and time
L = 1.;  		%  Lenth of the wire
T = 1.;			%  Final time


%  Parameters needed to solve the equation within the  Crank-Nicholson method
%  by using the LU decomposition
maxk = 2500;						%  Number of time steps
dt = T/maxk;
n = 50.;								%  Number of space steps
dx = L/n;
cond = 1./4.;						%  Conductivity
b = cond*dt/(dx*dx);		 		%  Parameter of the method 

%  Initial temperature of the wire: a sinus.
for i = 1:n+1
        x(i) =(i-1)*dx;
        v(i,1) =sin(pi*x(i));        
end
          
%  Temperature at the boundary (T=0)
for k=1:maxk+1
        v(1,k) = 0.;
        v(n+1,k) = 0.;
        time(k) = (k-1)*dt;
end

% Defining the Matrices M_right and M_left in the method
aal(1:n-2)=-b;
bbl(1:n-1)=2.+2.*b;
ccl(1:n-2)=-b;
MMl=diag(bbl,0)+diag(aal,-1)+diag(ccl,1);
[L,U]=lu(MMl);			% LU decomposition

aar(1:n-2)=b;
bbr(1:n-1)=2.-2.*b;
ccr(1:n-2)=b;
MMr=diag(bbr,0)+diag(aar,-1)+diag(ccr,1);


%  Implementation of the LU decomposition within the Crank-Nicholson method
for k=2:maxk		%  Time Loop   
   vv=v(2:n,k-1);
   qq=MMr*vv;
   w(1)=qq(1);
   for j=2:n-1
      w(j)=qq(j)-L(j,j-1)*w(j-1);
   end
   v(n,k)=w(n-1)/U(n-1,n-1);
   for i=n-1:-1:2
      v(i,k)=(w(i-1)-U(i-1,i)*v(i+1,k))/U(i-1,i-1);      
   end 
end

% Graphical representation of the temperature at different selected times
figure(1)
plot(x,v(:,1),'-',x,v(:,100),'-',x,v(:,300),'-',x,v(:,600),'-')
title('Temperature within the Crank-Nicholson method (LU)')
xlabel('X')
ylabel('T')

figure(2)
mesh(x,time,v')
title('Temperature within the Crank-Nicholson method (LU)')
xlabel('X')
ylabel('Temperature')
