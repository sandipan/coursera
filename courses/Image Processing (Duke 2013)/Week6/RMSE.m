function [rmse_value] = RMSE (IDEAL, NOISY)
% Compute the Root Mean Square Error (RMSE)
% Input: Noise-free original image IDEAL
%        NOISY image, possibly after applying a denoising method
% Images must be the same size.
% Images can be color or grayscale.

% Look up size of image (k=1 if grayscale).
[m,n,k] = size(IDEAL);

% Cast to double and vectorize.
IDEAL = double(IDEAL(:));
NOISY = double(NOISY(:));

% See lecture notes for RMSE formula.
rmse_value = 1/(m*n*k)*norm( IDEAL-NOISY, 'fro');

end

