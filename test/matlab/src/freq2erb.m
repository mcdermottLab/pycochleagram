% FUNCTION N_ERB = FREQ2ERB(FREQ_HZ)
%
% Converts Hz to ERBs, using the formula of Glasberg and Moore.

function n_erb = freq2erb(freq_Hz)

n_erb = 9.265*log(1+freq_Hz./(24.7*9.265));

