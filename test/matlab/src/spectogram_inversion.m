function wavSnd = invertSpectrogram(snd, P, N)
%P - Josh's texture params
%p - my params
%snd - sound cochleogram
%N - number of iterations - default 5

if nargin < 3
    N = 5;
end

ds_factor=P.audio_sr/P.env_sr; %factor by which envelopes are downsampled

%number of upsamples samples
nTotalSamples = ds_factor * size(snd, 2);

%initial noise vector
wavSnd = rand(1, nTotalSamples);

%create filters
%[audio_filts, ~] = make_erb_cos_filters(nTotalSamples, ...
%    P.audio_sr, P.N_audio_channels, P.low_audio_f, P.hi_audio_f);

%GENERATE FILTERS FOR SYNTHESIS
synth_dur_smp = nTotalSamples; %ceil(P.desired_synth_dur_s*P.audio_sr/ds_factor)*ds_factor; %ensures that length in samples is an integer multiple of envelope sr
%P.length_ratio = synth_dur_smp/length(orig_sound);

%make audio filters
if P.lin_or_log_filters==1 || P.lin_or_log_filters==2
    if P.use_more_audio_filters==0
        [audio_filts, audio_cutoffs_Hz] = make_erb_cos_filters(synth_dur_smp, P.audio_sr, P.N_audio_channels, P.low_audio_f, P.hi_audio_f);
    elseif P.use_more_audio_filters==1
        [audio_filts, audio_cutoffs_Hz] = make_erb_cos_filts_double2(synth_dur_smp, P.audio_sr, P.N_audio_channels, P.low_audio_f, P.hi_audio_f);
    elseif P.use_more_audio_filters==2
        [audio_filts, audio_cutoffs_Hz] = make_erb_cos_filts_quadruple2(synth_dur_smp, P.audio_sr, P.N_audio_channels, P.low_audio_f, P.hi_audio_f);
    end
elseif P.lin_or_log_filters==3 || P.lin_or_log_filters==4
    if P.use_more_audio_filters==0
        [audio_filts, audio_cutoffs_Hz] = make_lin_cos_filters(synth_dur_smp, P.audio_sr, P.N_audio_channels, P.low_audio_f, P.hi_audio_f);
    elseif P.use_more_audio_filters==1
        [audio_filts, audio_cutoffs_Hz] = make_lin_cos_filts_double(synth_dur_smp, P.audio_sr, P.N_audio_channels, P.low_audio_f, P.hi_audio_f);
    end
end

%decompress envelopes
if P.compression_option == 1
    snd = snd .^ (1 / P.comp_exponent);
elseif P.compression_option == 2
    snd = 10.^(snd) - P.log_constant;
end

%upsample envelopes via resample function
%upsampledEnv = zeros(nK, nTotalSamples);
%for i = 1:nK
%    upsampledEnv(i, :) = resample(snd(i, :), ds_factor, 1);
%end
%sum(upsampledEnv(:) < 0)
%upsampledEnv = upsampledEnv';

%upsample envelopes via interpolation
dT = 1 / ds_factor;
x = 1:length(snd);
xI = dT:dT:length(snd);
upsampledEnv = interp1(x, snd', xI);


for n = 1:N
    subband_phs = angle(hilbert(generate_subbands(wavSnd, audio_filts)));
    subband_sig = subband_phs .* upsampledEnv;
    subband_sig = real(subband_sig);
    subband_sig(isnan(subband_sig)) = 0;
    wavSnd = collapse_subbands(subband_sig, audio_filts);
    %plot(wavSnd); drawnow; pause(.5);
end
