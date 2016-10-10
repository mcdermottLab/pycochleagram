% [FILTS, HZ_CUTOFFS, FREQS] = MAKE_ERB_COS_FILTERS(SIGNAL_LENGTH, SR, N, LOW_LIM, HI_LIM)
%
% Returns N+2 filters as column vectors of FILTS 
% filters have cosine-shaped frequency responses, with center frequencies
% equally spaced on an ERB scale from LOW_LIM to HI_LIM
%
% Adjacent filters overlap by 50%.
%
% HZ_CUTOFFS is a vector of the cutoff frequencies of each filter. Because
% of the overlap arrangement, the upper cutoff of one filter is the center
% frequency of its neighbor.
%
% FREQS is a vector of frequencies the same length as FILTS, that can be
% used to plot the frequency response of the filters.
%
% There are N+2 filters because FILTS also contains lowpass and highpass
% filters to cover the ends of the spectrum.
%
% The squared frequency responses of the filters sums to 1, so that they
% can be applied once to generate subbands and then again to collapse the
% subbands to generate a sound signal, without changing the frequency
% content of the signal.
%
% Filters are to be applied multiplicatively in the frequency domain and
% thus have a length that scales with the signal length (SIGNAL_LENGTH).
%
% SR is the sampling rate
%
% intended for use with GENERATE_SUBBANDS and COLLAPSE_SUBBANDS

% Dec 2012 -- Josh McDermott <jhm@mit.edu>

function [filts,Hz_cutoffs,freqs] = make_erb_cos_filters(signal_length, sr, N, low_lim, hi_lim)

if rem(signal_length,2)==0 %even length
    nfreqs = signal_length/2;%does not include DC
    max_freq = sr/2;
    freqs = [0:max_freq/nfreqs:max_freq]; %go all the way to nyquist
else %odd length
    nfreqs = (signal_length-1)/2;
    max_freq = sr*(signal_length-1)/2/signal_length; %max freq is just under nyquist
    freqs = [0:max_freq/nfreqs:max_freq];
end   
cos_filts = zeros(nfreqs+1,N);

if hi_lim>sr/2
    hi_lim = max_freq;
end
%make cutoffs evenly spaced on an erb scale
cutoffs = erb2freq([freq2erb(low_lim) : (freq2erb(hi_lim)-freq2erb(low_lim))/(N+1) : freq2erb(hi_lim)]);

for k=1:N
    l = cutoffs(k);
    h = cutoffs(k+2); %adjacent filters overlap by 50%
    l_ind = min(find(freqs>l));
    h_ind = max(find(freqs<h));
    avg = (freq2erb(l)+freq2erb(h))/2;
    rnge = (freq2erb(h)-freq2erb(l));
    cos_filts(l_ind:h_ind,k) = cos((freq2erb( freqs(l_ind:h_ind) ) - avg)/rnge*pi); %map cutoffs to -pi/2, pi/2 interval
end

%add lowpass and highpass to get perfect reconstruction
filts = zeros(nfreqs+1,N+2);
filts(:,2:N+1) = cos_filts;
h_ind = max(find(freqs<cutoffs(2))); %lowpass filter goes up to peak of first cos filter
filts(1:h_ind,1) = sqrt(1 - filts(1:h_ind,2).^2);
l_ind = min(find(freqs>cutoffs(N+1))); %highpass filter goes down to peak of last cos filter
filts(l_ind:nfreqs+1,N+2) = sqrt(1 - filts(l_ind:nfreqs+1,N+1).^2);

Hz_cutoffs = cutoffs;
display(freq2erb(Hz_cutoffs'))

%subplot(2,1,1); plot(freqs,sum(filts.^2,2))
%subplot(2,1,2); semilogx(freqs,sum(filts.^2,2))
