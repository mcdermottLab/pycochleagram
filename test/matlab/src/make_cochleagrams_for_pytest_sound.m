function [sub_envs_1,sub_envs_2,sub_envs_3] = make_cochleagrams_for_pytest_sound(zp)

low_lim_human = 50;
high_lim_human = 20000;
N_human = floor(freq2erb(high_lim_human) - freq2erb(low_lim_human)) - 1;

IN_DIR = uigetdir();
OUT_DIR = uigetdir();
FN_ZERO_PAD = ['%0', num2str(ceil(log10(N_human + 1))), 'd'];
sounds = dir(fullfile(IN_DIR,'*.wav'));


for k = 1:2%length(sounds)
    
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
  
  
  subbands_1 = generate_subbands(wavData, filts_1); % filters signal (ct) with generated audiobank filter
  subbands_2 = generate_subbands(wavData, filts_2);
  subbands_4 = generate_subbands(wavData, filts_4);
  
  if zp==1
    subbands_1 = subbands_1(1:end/2,:);
    subbands_2 = subbands_2(1:end/2,:);
    subbands_4 = subbands_4(1:end/2,:);
  end
  
  sub_envs_1 = abs(hilbert(subbands_1));
  sub_envs_2 = abs(hilbert(subbands_2));
  sub_envs_4 = abs(hilbert(subbands_4));
  
  
  figure; imagesc(flipud(sub_envs_1'));
  wfn = fullfile(OUT_DIR, ['human_subands_test_', num2str(k, FN_ZERO_PAD),'.mat' ]);
  display(['saving output to: ', wfn]);
  save(wfn);
end
    
end
