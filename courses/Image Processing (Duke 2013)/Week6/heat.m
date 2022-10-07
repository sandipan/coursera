function [u] = heat (f)
%Isotropic heat equation
%Written in lecture 10/7/2014
%Input: Initial temperature distribution f
%Output: Solution of the heat equation u

%Parameters
dt = 0.1;  %Time step
T = 20;  %Stopping time
K = 1;  %Conductivity

%Initialize u=f.  Convert to double so we can do arithmetic.
u = double(f);
[m,n] = size(u);

for t = 0:dt:T
   u_xx = u(:,[2:n,n]) - 2*u + u(:,[1,1:n-1]);
   u_yy = u([2:m,m],:) - 2*u + u([1,1:m-1],:);
   u = u + K * dt * (u_xx + u_yy);
   imshow(uint8(u)); 
   title(['t=',num2str(t)]);
   drawnow;  %Matlab will not draw iterates without this command.
end

u = uint8(u);

end

