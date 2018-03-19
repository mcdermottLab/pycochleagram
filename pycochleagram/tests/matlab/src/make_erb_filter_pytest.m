OUT_DIR = '../../data/outputs/'
MODE = 'rand'
N_STIM = 200
FN_ZERO_PAD = ['%0', num2str(ceil(log10(N_STIM + 1))), 'd'];

signal_length_range = [1 220501];
sr_range = [10000 44101];
low_lim_range = [0 500];
hi_lim_range = [10000 20000];
low_erb = freq2erb(low_lim_range(1));
hi_erb = freq2erb(hi_lim_range(2));
N_max = floor(hi_erb - low_erb) - 1; % find the number of filters to make (specific to human)
N_range = [1 N_max];

% make output directory if necessary
if ~exist(OUT_DIR, 'dir')
    mkdir(OUT_DIR)
end

% generate the parameter grid, if necessary 
if strcmpi(MODE, 'grid')
    % expand [1, 2] array of interval endpoints to a range
    expand = @(x) x(1):x(2);
    % create a parameter grid of all possible parameter choices
    p_sl, p_sr, p_N, p_l, p_h = ndgrid(expand(signal_length), expand(sr_range), expand(N_range),...
                        expand(low_lim_range), expand(hi_lim_range));
    rowflat = @(x) reshape(x, 1, []);
    % flatten parameter grid (for easier inspection)
    param_grid = [rowflat(p_sl); rowflat(p_sr); rowflat(p_N); rowflat(p_l); rowflat(p_h)];
    % clean up
    clear p_sl p_sr p_N p_l p_h
end

stim_ctr = 0;
while stim_ctr <= N_STIM
    stim_ctr = stim_ctr + 1;

    if strcmpi(MODE, 'rand')
        % get random parameter set
        signal_length = randi(signal_length_range);
        sr = randi(sr_range);
        N = randi(N_range);
        low_lim = randi(low_lim_range);
        hi_lim = randi(hi_lim_range);
    elseif strcmpi(MODE, 'grid')
        % get parameter set from grid
        temp_param = param_grid(stim_ctr);
        % unpack the params for this iteration
        [signal_length, sr, N, low_lim, hi_lim] = feval(@(x)x{:}, num2cell(temp_param));
    else
        error(['Unrecognized MODE: ', MODE]);
    end
    
    % generate filter banks at 1x, 2x, 4x
    [filts_1, Hz_cutoffs_1, freqs_1] = make_erb_cos_filters(signal_length, sr, N, low_lim, hi_lim);
    [filts_2, Hz_cutoffs_2, freqs_2] = make_erb_cos_filts_double2(signal_length, sr, N, low_lim, hi_lim);
    [filts_4, Hz_cutoffs_4, freqs_4] = make_erb_cos_filts_quadruple2(signal_length, sr, N, low_lim, hi_lim);

    % save all variables to a file
    wfn = fullfile(OUT_DIR, ['erb_human_filters_test_', num2str(stim_ctr, FN_ZERO_PAD),'.mat' ]);
    display(['saving output to: ', wfn]);
    save(wfn, '-regexp', '^(?!(param_grid)$).')
end
