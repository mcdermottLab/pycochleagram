% SIGNAL = COLLAPSE_SUBBANDS(SUBBANDS, FILTS)
%
% filters SUBBANDS with FILTS and then sums them up to generate a full
% bandwidth signal
%
% SUBBANDS would typically be generated with GENERATE_SUBBANDS, then
% possibly modified
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

function signal = collapse_subbands(subbands, filts)

signal_length = length(subbands);
filt_length = size(filts,1);
if rem(signal_length,2)==0 %even length -
    fft_filts = [filts' fliplr(filts(2:filt_length-1,:)')]'; %generate negative frequencies in right place; filters are column vectors
else %odd length
    fft_filts = [filts' fliplr(filts(2:filt_length,:)')]';
end
fft_subbands = fft_filts.*(fft(subbands));
%subbands = real(ifft(fft_subbands));
subbands = ifft(fft_subbands);
signal = sum(subbands,2);
