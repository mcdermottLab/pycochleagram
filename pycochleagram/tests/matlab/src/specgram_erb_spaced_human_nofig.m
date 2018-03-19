%generates spectrogram with ERB-spaced frequency axis for sound <s>
%
%this version makes filters inside the function
%
%if zp is 1, signal is zero-padded
% name is the name of figure to save to
function img = specgram_erb_spaced_human(s,sr,low_lim,hi_lim,zp)

if zp==1
    if size(s,1)==1%row vector
        s = [s zeros(size(s))];
    elseif size(s,2)==1%column vector
        s = [s; zeros(size(s))];
    end
end

signal_length=length(s);

% find the number of filters to make - this will be specific to human
N = floor(freq2erb(hi_lim) - freq2erb(low_lim)) - 1;

%%% DEBUG %%%
% [filts,Hz_cutoffs,freqs] = make_erb_cos_filts_double2(length(s), sr, 3, low_lim, hi_lim); % make audiobank filter
%%%%%
[filts,Hz_cutoffs,freqs] = make_erb_cos_filts_double2(length(s), sr, N, low_lim, hi_lim); % make audiobank filter

% filts = filts(:,2:end-1); %throw out low and high end (make sure to take into account if it was filts were made with double quadruple!
filts = filts(:,3:end-2); % using 2x oversampled filter, so throw out 2 lowpass and 2 highpass filters

subbands = generate_subbands(s,filts);
if zp==1
    subbands = subbands(1:end/2,:);
    signal_length = signal_length/2;
end

thresh = -60;
env_sr = 6000;


sub_envs = abs(hilbert(subbands));

sub_envs = resample(sub_envs,env_sr,sr);

sub_envs(find(sub_envs<0)) = eps;

sub_envs = 20*log10(sub_envs/max(max(sub_envs)));

%sub_envs(find(sub_envs<thresh)) = thresh;

blur_level = 1;

%figure;%('Position', [180 55 670 1042],'PaperPosition',[0.25 0.25 8 10.5]);
%figure('Visible','off');
%imagesc(upBlur(flipud(sub_envs'),blur_level));colormap(flipud(gray));

img = flipud(sub_envs');

%figure; imagesc(flipud(sub_envs'));colormap(flipud(gray)); % plot subbands

%ylabel('Frequency (Hz)','FontSize',10);
if N<=40
    tick_spacing = 5;
else
    tick_spacing = 10;
end
%set(gca,'YTick',([0:tick_spacing:floor(N/tick_spacing)*tick_spacing]+1)*2^blur_level,'YTickLabel',num2str(round(Hz_cutoffs([N+1:-tick_spacing:(N+1-floor(N/tick_spacing)*tick_spacing)]))'));

if length(sub_envs)/env_sr < 2
    time_spacing = .4;
    %set(gca,'XTick',[0:time_spacing:floor(length(sub_envs)/env_sr/time_spacing)*time_spacing]*2*env_sr,'XTickLabel',num2str(1000*[0:time_spacing:floor(length(sub_envs)/env_sr/time_spacing)*time_spacing]'));
    %xlabel('Time (ms)','FontSize',10);    
else
    time_spacing = 1;
    %set(gca,'XTick',[0:time_spacing:floor(length(sub_envs)/env_sr/time_spacing)*time_spacing]*2*env_sr,'XTickLabel',num2str([0:time_spacing:floor(length(sub_envs)/env_sr/time_spacing)*time_spacing]'));
    %xlabel('Time (sec)','FontSize',10);
end

%set(gca,'CLim',[-100 0]);
%colorbar

