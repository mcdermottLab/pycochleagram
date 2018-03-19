function [subbands, fft_filts, fft_sample, fft_subbands] = generate_subbands_debug(signal, filts)

% SUBBANDS = GENERATE_SUBBANDS(SIGNAL, FILTS)
%
% filters SIGNAL with FILTS to generate subbands DeBUG VERSION FOR
% RETURNING ALL VARIABLES
%
% FILTS would typically be generated with one of the MAKE...FILTERS
% functions, e.g. MAKE_ERB_COS_FILTERS. FILTS contains coefficients for
% positive frequencies and is applied multiplicatively in the frequency
% domain
%
% FILTS are typically generated so that the sum of their squared response
% is flat across the frequency spectrum. They are thus applied once to
% generate subbands and then once more to collapse the subbands into a full
% bandwidth signal. This has the advantage of ensuring that the subbands
% remain appropriately bandlimited following any modifications that are
% made.
%

% Dec 2012 -- Josh McDermott <jhm@mit.edu>

if size(signal,1)==1 %turn into column vector
    signal = signal';
end
N=size(filts,2)-2;
signal_length = length(signal);
filt_length = size(filts,1);
fft_sample = fft(signal);
if rem(signal_length,2)==0 %even length - 
    fft_filts = [filts' fliplr(filts(2:filt_length-1,:)')]'; %generate negative frequencies in right place; filters are column vectors
else %odd length
    fft_filts = [filts' fliplr(filts(2:filt_length,:)')]';
end
% size(fft_filts)
fft_subbands = fft_filts.*(fft_sample*ones(1,N+2));%multiply by array of column replicas of fft_sample
subbands = real(ifft(fft_subbands)); %ifft works on columns; imag part is small, probably discretization error?