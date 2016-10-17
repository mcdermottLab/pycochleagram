function [sub_envs_1,sub_envs_2,sub_envs_3] = make_cochleagrams_for_pytest_sound(zp)

low_lim_human = 50;
high_lim_human = 20000;
N_human = floor(freq2erb(high_lim_human) - freq2erb(low_lim_human)) - 1;

if strcmp(getenv('USER'), 'raygon')
    IN_DIR = '/Users/raygon/Desktop/mdlab/projects/cochleagram/test/data/input/timit/';
    OUT_DIR = '/Users/raygon/Desktop/mdlab/projects/cochleagram/test/data/output/coch_human/';
else
    IN_DIR = uigetdir();
    OUT_DIR = uigetdir();
end

FN_ZERO_PAD = ['%0', num2str(ceil(log10(N_human + 1))), 'd'];
sounds = dir(fullfile(IN_DIR,'*.wav'));


for k = 1:100%length(sounds)
    
    % reading in data
    baseFileName = sounds(k).name;
    fullFileName = fullfile(IN_DIR, baseFileName);
    fprintf(1, 'Reading %s\n', fullFileName);
    [wavData, Fs] = audioread(fullFileName);
    disp(size(wavData))
    
    if zp==1
        if size(wavData,1)==1%row vector
            wavData = [wavData zeros(size(wavData))];
        elseif size(wavData,2)==1%column vector
            wavData = [wavData; zeros(size(wavData))];
        end
    end
    
    fprintf(1, 'Creating filters for %s\n', fullFileName);
    % generate filter banks at 1x, 2x, 4x
    [filts_1, Hz_cutoffs_1, freqs_1] = make_erb_cos_filters(length(wavData), Fs, N_human, low_lim_human, high_lim_human);
    [filts_2, Hz_cutoffs_2, freqs_2] = make_erb_cos_filts_double2(length(wavData), Fs, N_human, low_lim_human, high_lim_human);
    [filts_4, Hz_cutoffs_4, freqs_4] = make_erb_cos_filts_quadruple2(length(wavData), Fs, N_human, low_lim_human, high_lim_human);
    
%     % make full filter set (the output of make_erb_cos_filts is only half)
%     signal_length = length(wavData);
%     full_filts_1 = make_full_filter_set(filts_1, signal_length);
%     full_filts_2 = make_full_filter_set(filts_2, signal_length);
%     full_filts_4 = make_full_filter_set(filts_4, signal_length);

    % gener
    [subbands_1, fft_filts_1, fft_sample_1, fft_subbands_1] = generate_subbands_debug(wavData, filts_1); % filters signal (ct) with generated audiobank filter
    [subbands_2, fft_filts_2, fft_sample_2, fft_subbands_2] = generate_subbands_debug(wavData, filts_2);
    [subbands_4, fft_filts_4, fft_sample_4, fft_subbands_4] = generate_subbands_debug(wavData, filts_4);
    
    if zp==1
        subbands_1 = subbands_1(1:end/2,:);
        subbands_2 = subbands_2(1:end/2,:);
        subbands_4 = subbands_4(1:end/2,:);
    end
    
    sub_envs_1 = abs(hilbert(subbands_1));
    sub_envs_2 = abs(hilbert(subbands_2));
    sub_envs_4 = abs(hilbert(subbands_4));
    
    
%     figure; imagesc(flipud(sub_envs_1'));
    wfn = fullfile(OUT_DIR, ['human_subands_test_', num2str(k, FN_ZERO_PAD),'.mat' ]);
    display(['saving output to: ', wfn]);
    save(wfn);
end

end

function fft_filts = make_full_filter_set(filts, signal_length)
    % Use this to conver the output of make_erb_cos_filts to the full
    % filterbank to apply to the fft of a signal
    filt_length = size(filts,1);
    if rem(signal_length,2)==0 %even length -
        fft_filts = [filts' fliplr(filts(2:filt_length-1,:)')]'; %generate negative frequencies in right place; filters are column vectors
    else %odd length
        fft_filts = [filts' fliplr(filts(2:filt_length,:)')]';
    end
end
