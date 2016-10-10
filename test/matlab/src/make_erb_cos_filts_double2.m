
% [FILTS,HZ_CUTOFFS,FREQS] = MAKE_ERB_COS_FILTS_DOUBLE2(SIGNAL_LENGTH, SR,
% N, LOW_LIM, HI_LIM)
%
% Returns 2*N+5 filters as column vectors of FILTS 
% filters have cosine-shaped frequency responses, with center frequencies
% equally spaced on an ERB scale from LOW_LIM to HI_LIM
%
% This function returns a filterbank that is 2x overcomplete compared to
% MAKE_ERB_COS_FILTS (to get filterbanks that can be compared with each
% other, use the same value of N in both cases). Adjacent filters overlap
% by 75%.
%
% As in MAKE_ERB_COS_FILTS, FILTS also contains lowpass
% and highpass filters to cover the ends of the spectrum.
%
% HZ_CUTOFFS is a vector of the cutoff frequencies of each filter. 
%
% FREQS is a vector of frequencies the same length as FILTS, that can be
% used to plot the frequency response of the filters.
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
%
% May 2014 -- modified to correct erb2freq function call -- Josh McDermott


function [filts,Hz_cutoffs,freqs] = make_erb_cos_filts_double2(signal_length, sr, N, low_lim, hi_lim)

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

num_filters = 2*N+1;
%make cutoffs evenly spaced on an erb scale
spacing = (freq2erb(hi_lim)-freq2erb(low_lim))/(num_filters+1);%in ERBs
center_freqs = linspace(freq2erb(low_lim)+spacing, freq2erb(hi_lim)-spacing, num_filters); %in ERBs

% display(center_freqs)
for k=1:num_filters
    l = erb2freq(center_freqs(k)-2*spacing);
    h = erb2freq(center_freqs(k)+2*spacing);
%     display(['k: ', num2str(k), ', l h: [', num2str(l), ', ', num2str(h), ']'])
    l_ind = min(find(freqs>l));
    h_ind = max(find(freqs<h));
    avg = (freq2erb(l)+freq2erb(h))/2;
    rnge = (freq2erb(h)-freq2erb(l));
    cos_filts(l_ind:h_ind,k) = cos((freq2erb( freqs(l_ind:h_ind) ) - avg)/rnge*pi); %map cutoffs to -pi/2, pi/2 interval
end

%add lowpass and highpass to get perfect reconstruction
filts = zeros(nfreqs+1,num_filters+4);
filts(:,3:num_filters+2) = cos_filts;
%lowpass filters go up to peaks of first, second cos filters
h_ind = max(find(freqs<erb2freq(center_freqs(1))));
filts(1:h_ind,1) = sqrt(1 - filts(1:h_ind,3).^2);
h_ind = max(find(freqs<erb2freq(center_freqs(2))));
filts(1:h_ind,2) = sqrt(1 - filts(1:h_ind,4).^2);
%highpass filters go down to peaks of last two cos filters
l_ind = min(find(freqs>erb2freq(center_freqs(num_filters))));
filts(l_ind:nfreqs+1,num_filters+4) = sqrt(1 - filts(l_ind:nfreqs+1,num_filters+2).^2);
l_ind = min(find(freqs>erb2freq(center_freqs(num_filters-1))));
filts(l_ind:nfreqs+1,num_filters+3) = sqrt(1 - filts(l_ind:nfreqs+1,num_filters+1).^2);

filts = filts/sqrt(2); %so that squared freq response adds to 1

center_freqs = erb2freq([center_freqs(1)-2*spacing center_freqs(2)-2*spacing center_freqs center_freqs(num_filters-1)+2*spacing center_freqs(num_filters)+2*spacing]);
Hz_cutoffs = center_freqs;
% display(freq2erb(Hz_cutoffs)')
Hz_cutoffs(find(Hz_cutoffs<0)) = 1;

%subplot(2,1,1); plot(freqs,sum(filts.^2,2))
%subplot(2,1,2); semilogx(freqs,sum(filts.^2,2))
