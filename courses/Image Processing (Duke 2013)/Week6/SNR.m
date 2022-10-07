function [snr_value] = SNR (IDEAL, NOISY)
% Compute the Signal-to-Noise Ratio (SNR)
% Input: Noise-free original image IDEAL
%        NOISY image, possibly after applying a denoising method
% Images must be the same size.
% Images can be color or grayscale.

% Cast to double and vectorize.
IDEAL = double(IDEAL(:));
NOISY = double(NOISY(:));

% See lecture notes for SNR formula.
snr_value = 20*log( norm(IDEAL,'fro') / norm(IDEAL-NOISY,'fro') );

end

