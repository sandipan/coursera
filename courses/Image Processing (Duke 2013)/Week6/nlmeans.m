function [g] = nlmeans (f, M)

% Non-Local Means denoising
%
% Input: Noisy grayscale image f
%        Window size M (M odd)
% Output: Denoised grayscale image g
%
% This version only denoises the interior of the image, could
%  have used reflective or periodic boundary conditions instead.
% The implementation here is very inefficient, but it is
%  meant to be as straightforward as possible.
% We can speed it up considerably by replacing the loops
%  with matrix multiplications.

%First check that image is grayscale and cast to double.
[m,n,num_colors] = size(f);
if num_colors>1
   f = rgb2gray(f);
end;
f = double(f);
g = zeros(size(f));

%Parameter h is very important.  Needs to be set experimentally.
%I used sqrt(2)*sigma where sigma is standard deviation of the image.
h = sqrt(2)*std(f(:));

%Prepare list of neighborhoods N.
%Each row of N is a MxM neighborhood from the image interior.
M_half = (M-1)/2;  %M is odd, so this tells us half the window size.
num = 1;
for i = 1+M_half : m-M_half
   for j=1+M_half : n-M_half
      N(num,:) = reshape( f(i-M_half:i+M_half , j-M_half:j+M_half), [1,M*M] );
      num = num + 1;
   end;
end;

%Now for each pixel in the image, we compare it to each row of N.
%We compute the weight for each neighborhood comparison.
%Plots the image as it goes, which slows down the program.
for i = 1+M_half : m-M_half
   for j=1+M_half : n-M_half
      current_nbhd = reshape( f(i-M_half:i+M_half , j-M_half:j+M_half), [1,M*M] );
      Z = 0;   %Normalization constant.
      for k = 1:num-1
         w = exp(- (norm(current_nbhd - N(k,:))^2) / (h^2) );
         g(i,j) = g(i,j) + N(k,(M^2+1)/2)*w;  %Pixel in middle of neighborhood is at position (M^2+1)/2.
         Z = Z + w;
      end;
      g(i,j) = g(i,j) / Z;  %Divide by normalization constant.

      %Draw results
      subplot(121); imagesc(f); title('Original');
      subplot(122); imagesc(g, [min(f(:)),max(f(:))]); title(['NL-Means Pixel i=',num2str(i),' j=',num2str(j)]);
      colormap gray;
      drawnow;
   end;
end;

g = uint8(g);