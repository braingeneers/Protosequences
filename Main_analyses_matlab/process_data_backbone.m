%% load data

% load data
load('t_spk_mat_sorted.mat')

% make empty result array for shuffled datasets
t_spk_mat_all_shuf = [];

% get list of all files in the directory
all_files = dir;

% for each file
for file = 1:length(all_files)
    
    % if the file contains the right substring
    if contains(all_files(file).name, 't_spk_mat_sorted_shuff')
        
        % load data
        shuf_cont = load(all_files(file).name);
        
        if isfield(shuf_cont, 'all_shuff')
        
            % store loaded data in result array
            t_spk_mat_all_shuf = cat(3, t_spk_mat_all_shuf, permute(shuf_cont.all_shuff, [3,2,1]));
            
        end % if
    end % if
end % file

%% Compute population acitivity

% define parameters
TIME_RANGE = [0,300];% analysis range in s
THR_BURST = 4; % threshold for population peak detection
MIN_BURST_DIFF = 800; % minimum time between bursts (ms)
SQUARE_WIDTH = 20; % square window size for coarse population rate
GAUSS_SIGMA = 100; % gaussian window size for coarse population rate
SQUARE_WIDTH_ACC = 5; % square window size for specific population rate
GAUSS_SIGMA_ACC = 5; % gaussian window size for specific population rate
BURST_EDGE_MULT_THRESH = 0.1; % population rate threshold (percentage of peak) for burst edge detections
CLEAR_INTERMEDIATE = true; % clears redundant data after processing

% smooth summed spike times with square window
square_smooth_summed_spike = smoothdata(sum(t_spk_mat, 2),'movmean',SQUARE_WIDTH);

% smooth smoothed spike times with gaussian window
pop_rate = smoothdata(square_smooth_summed_spike,'gaussian',GAUSS_SIGMA);

% compute rms of pop rate
pop_rms = rms(pop_rate);

% detect peaks
[peak_amp, peaks] = findpeaks(pop_rate, 'MinPeakHeight', pop_rms*THR_BURST, 'MinPeakDistance', MIN_BURST_DIFF);

% remove peaks larger than TIME_RANGE
peak_amp(peaks>TIME_RANGE(2)*1000) = [];
peak_amp(peaks<TIME_RANGE(1)*1000) = [];
peaks(peaks>TIME_RANGE(2)*1000) = [];
peaks(peaks<TIME_RANGE(1)*1000) = [];

% smooth summed spike times with square window
square_smooth_summed_spike_acc = smoothdata(sum(t_spk_mat, 2),'movmean',SQUARE_WIDTH_ACC);

% smooth smoothed spike times with gaussian window
pop_rate_acc = smoothdata(square_smooth_summed_spike_acc,'gaussian',GAUSS_SIGMA_ACC);

% make empty result arrays
edges = NaN(length(peaks), 2);
tburst = NaN(1,length(peaks));

% for each detected burst
for burst = 1:length(peaks)
    
    % find all frames with network activity below burst threshold
    frames_below_thresh = find(pop_rate < peak_amp(burst)*BURST_EDGE_MULT_THRESH);
    
    % compute time of frames relative to burst peak
    rel_frames = peaks(burst) - frames_below_thresh;  % negative values means after burst
    
    % find smallest relative frame that occurs before the burst
    [rel_burst_start, ~] = min(rel_frames(rel_frames > 0));
    
    % find smallest relative frame that occurs before the burst
    [rel_burst_end, ~] = max(rel_frames(rel_frames < 0));
    
    % store burst edge results
    edges(burst,:) = [peaks(burst)-rel_burst_start, peaks(burst)-rel_burst_end];
    
    % find peak between edges
    [~, acc_peak] = max(pop_rate_acc(edges(burst,1):edges(burst,2)));
    
    % store burst peak 
    tburst(burst) = acc_peak+edges(burst,1);

end % burst

% remove bursts too far on the edge of the recording
if tburst(1)<=450
    tburst(1) = [];
    edges(1,:) = [];
end

if min(size(t_spk_mat,1), TIME_RANGE(2)*1000)-tburst(end)<=1000
    tburst(end) = [];
    edges(end,:) = [];
end


%% compute backbone units

MIN_SPIKES = 2; % define minimum number of spikes per burst
FRAC_THRESH = 1; % define minimum fraction of bursts

% initiate result cell array
spikes_per_burst = zeros(size(t_spk_mat,2), size(edges,1));
 
% for each unit
for unit = 1:size(t_spk_mat,2)
    
    % obtain spike times in ms
    unit_spk_times = find(t_spk_mat(:,unit));
    
    % for each burst
    for burst = 1:size(edges,1)
        
        % obtain all spike times within burst
        burst_times = unit_spk_times(unit_spk_times >= edges(burst,1) & unit_spk_times <= edges(burst,2));
        
        % store number of spikes in burst
        spikes_per_burst(unit, burst) = length(burst_times);
        
    end % burst
end % unit

% determine bursts above MIN_SPIKES
above_thresh = spikes_per_burst >= MIN_SPIKES;

% % compute fraction of bursts above threshold per unit
frac_per_unit = sum(above_thresh, 2)/size(edges,1);
frac_per_burst = sum(above_thresh, 1)/size(t_spk_mat,2);

% % store scaffold and non scaffold units
scaf_units = find(frac_per_unit >= FRAC_THRESH);
non_scaf_units = find(frac_per_unit < FRAC_THRESH);


%% calculate firing rates per unit

% define parameters
GAUSS_SIGMA_SU = 50; % sigma for gaussian smoothing in ms

% obtain t_spk_mat for bursts only
t_spk_mat_burst_only = remove_spk_outside_burst(t_spk_mat, edges);
t_spk_mat_burst_only_all_shuf = remove_spk_outside_burst(t_spk_mat_all_shuf, edges);

% count number of spikes per unit
spk_count = sum(t_spk_mat);

% compute firing rate per unit
[rate_mat, spk_times, spk_times_id] = compute_rate_isi(t_spk_mat, GAUSS_SIGMA_SU, TIME_RANGE);
[rate_mat_burst_only, ~, ~] = compute_rate_isi(t_spk_mat_burst_only, GAUSS_SIGMA_SU, TIME_RANGE);
[rate_mat_all_shuf, spk_times_all_shuf, spk_times_id_all_shuf] = compute_rate_isi(t_spk_mat_all_shuf, GAUSS_SIGMA_SU, TIME_RANGE);
[rate_mat_burst_only_all_shuf, ~, ~] = compute_rate_isi(t_spk_mat_burst_only_all_shuf, GAUSS_SIGMA_SU, TIME_RANGE);

% store first of shuffled data for plotting example
t_spk_mat_rand = t_spk_mat_all_shuf(:,:,1);
rate_mat_burst_only_rand = rate_mat_burst_only_all_shuf(:,:,1);
rate_mat_rand = rate_mat_all_shuf(:,:,1);
spk_times_rand = spk_times_all_shuf(:,1);
spk_times_id_rand = spk_times_id_all_shuf(:,1);

if CLEAR_INTERMEDIATE
    clearvars spk_times_all_shuf spk_times_id_all_shuf
end

%% compute firing rate peaks and average signals per burst
 
% specify plot parameters
PEAK_THRESH = 1000; % minimum peak firing rate
MIN_SPIKES = 2;
CUT_RANGE = [250,500]; % time before and after burst peak to plot (ms)

% compute burst signal metrics per burst
[act_times, av_rate, cut_spk_mat] = compute_sig_per_burst(rate_mat, ...
    t_spk_mat, edges, tburst, PEAK_THRESH, MIN_SPIKES, CUT_RANGE);
[act_times_all_shuf, av_rate_all_shuf, cut_spk_mat_all_shuf] = compute_sig_per_burst(rate_mat_all_shuf, ...
    t_spk_mat_all_shuf, edges, tburst, PEAK_THRESH, MIN_SPIKES, CUT_RANGE);

if CLEAR_INTERMEDIATE
    clearvars t_spk_mat_all_shuf rate_mat_all_shuf
end

%%  compute reordering of units based on scaf and median act time

% set sort metric to median of act times for bursts with at least 2 spikes
act_times_copy = act_times;
act_times_copy(above_thresh==0) = NaN;
sort_metric = median(act_times_copy, 'omitnan');

% define scaf window
scaf_window = [min(sort_metric), max(sort_metric)];

% define burst window
burst_window = [min(edges(:,1)-tburst'), max(edges(:,2)-tburst')];

% make copies of the data for scaf and nonscaf units
max_i_mean_scaf = sort_metric;
max_i_mean_non_scaf = sort_metric;

% select only scaffold or nonscaffold data
max_i_mean_scaf(non_scaf_units) = NaN;
max_i_mean_non_scaf(scaf_units) = NaN;

% sort results based on peak time
[~,sort_i_mean_scaf] = sort(max_i_mean_scaf);
[~,sort_i_mean_non_scaf] = sort(max_i_mean_non_scaf);

% store results
mean_rate_ordering = fliplr([sort_i_mean_scaf(1:length(scaf_units)), sort_i_mean_non_scaf(1:length(non_scaf_units))]);
   

%% compute correlation scores burst to burst

% set parameters
MAXLAG_BTB = 10;
WINDOW = [250,500];
MIN_SPIKES = 2;
MIN_BURST_FRAC = 0.3;

% compute burst to burst correlations for normal data
[all_btb_corr_scores, av_btb_corr_scores] = compute_burst_to_burst_corr(rate_mat_burst_only, ...
    t_spk_mat_burst_only, tburst, MAXLAG_BTB, WINDOW, MIN_SPIKES, MIN_BURST_FRAC);
 
% compute burst to burst correlations for randomized data
[all_btb_corr_scores_all_shuf, av_btb_corr_scores_all_shuf] = compute_burst_to_burst_corr(rate_mat_burst_only_all_shuf, ...
    t_spk_mat_burst_only_all_shuf, tburst, MAXLAG_BTB, WINDOW, MIN_SPIKES, MIN_BURST_FRAC);

% compute average and std over all shuffled datasets
all_btb_corr_scores_av_shuf = mean(all_btb_corr_scores_all_shuf, 4, 'omitnan');
av_btb_corr_scores_av_shuf = mean(av_btb_corr_scores_all_shuf, 3, 'omitnan');
all_btb_corr_scores_std_shuf = std(all_btb_corr_scores_all_shuf, [], 4, 'omitnan');
av_btb_corr_scores_std_shuf = std(av_btb_corr_scores_all_shuf, [], 3, 'omitnan');

if CLEAR_INTERMEDIATE
    clearvars all_btb_corr_scores_all_shuf av_btb_corr_scores_all_shuf t_spk_mat_burst_only_all_shuf
end

%% compute pairwise cross correlation scores

% specify max lag
MAXLAG_PW = 350;

% compute pairwise cross correlations for normal data
[all_pw_corr_vals, all_pw_corr_lags] = compute_pairwise_corr(rate_mat_burst_only, MAXLAG_PW);

% compute pairwise cross correlations for randomized data
[all_pw_corr_vals_all_shuf, all_pw_corr_lags_all_shuf] = compute_pairwise_corr(rate_mat_burst_only_all_shuf, MAXLAG_PW);

% save results for all shuffled datasets
save('pw_corr_all_shuf', 'all_pw_corr_vals_all_shuf', 'all_pw_corr_lags_all_shuf', '-v7.3')
'pw corr results for all shuffled saved'

% compute average and std over all shuffled datasets
all_pw_corr_vals_av_shuf = mean(all_pw_corr_vals_all_shuf, 3, 'omitnan');
all_pw_corr_lags_av_shuf = mean(all_pw_corr_lags_all_shuf, 3, 'omitnan');
all_pw_corr_vals_std_shuf = std(all_pw_corr_vals_all_shuf, [], 3, 'omitnan');
all_pw_corr_lags_std_shuf = std(all_pw_corr_lags_all_shuf, [], 3, 'omitnan');

if CLEAR_INTERMEDIATE
    clearvars all_pw_corr_lags_all_shuf all_pw_corr_vals_all_shuf
end

%% compute burst similarity score

% specify time relative to burst peak to analyze
FRAME_RANGE = [-450:1000];

% compute burst similarity
[mean_cos_sim, cos_sim, centered_pop_rate] = comp_burst_sim(rate_mat, pop_rate, FRAME_RANGE, tburst);
[mean_cos_sim_all_shuf, cos_sim_all_shuf, ~] = comp_burst_sim(rate_mat_burst_only_all_shuf, pop_rate, FRAME_RANGE, tburst);
[mean_cos_sim_scaf, cos_sim_scaf, ~] = comp_burst_sim(rate_mat(:,scaf_units), pop_rate, FRAME_RANGE, tburst);
[mean_cos_sim_non_scaf, cos_sim_non_scaf, ~] = comp_burst_sim(rate_mat(:,non_scaf_units), pop_rate, FRAME_RANGE, tburst);

% compute average and std over all shuffled datasets
mean_cos_sim_av_shuf = mean(mean_cos_sim_all_shuf, 2, 'omitnan');
cos_sim_av_shuf = mean(cos_sim_all_shuf, 4, 'omitnan');
mean_cos_sim_std_shuf = std(mean_cos_sim_all_shuf, [], 2, 'omitnan');
cos_sim_std_shuf = std(cos_sim_all_shuf, [], 4, 'omitnan');

if CLEAR_INTERMEDIATE
    clearvars mean_cos_sim_all_shuf cos_sim_all_shuf
end

%% perform PCA analysis

norm_method = 'scale';

% compute manifolds
[sbsc, vars, contributions] = compute_manifolds(rate_mat_burst_only, scaf_units, non_scaf_units, norm_method);
[sbsc_all_shuf, vars_all_shuf, contributions_all_shuf] = compute_manifolds(rate_mat_burst_only_all_shuf, scaf_units, non_scaf_units, norm_method);

% compute times relative to closest burst peak per frame (for plotting)
peak_rel = compute_trel_brst_peak(tburst, burst_window, (TIME_RANGE(2)-TIME_RANGE(1))*1000);


%% save results

% save results for normal data
save('single_recording_metrics', 'edges', 'tburst', 'pop_rate', 'spk_count', 't_spk_mat', ... 
    'above_thresh', 'frac_per_unit', 'frac_per_burst', 'scaf_units', ...
    'non_scaf_units', 'act_times', 'av_rate', 'cut_spk_mat', ...
    'mean_rate_ordering', 'rate_mat', 'rate_mat_burst_only', 'spk_times', ...
    'spk_times_id', 'all_btb_corr_scores', 'av_btb_corr_scores', ...
    'all_pw_corr_vals', 'all_pw_corr_lags', 'burst_window', 'scaf_window', ...
    'mean_cos_sim', 'cos_sim', 'mean_cos_sim_scaf', 'cos_sim_scaf', 'mean_cos_sim_non_scaf', ...
    'cos_sim_non_scaf', 'centered_pop_rate', 'sbsc', 'vars', 'contributions', ...
    '-v7.3')



% save results for randomized data
save('single_recording_metrics_shuff', 'edges', 'tburst', 'pop_rate', 'spk_count', 't_spk_mat_rand', ... 
    'above_thresh', 'frac_per_unit', 'frac_per_burst', 'scaf_units', ...
    'non_scaf_units', 'act_times_all_shuf', 'av_rate_all_shuf', 'cut_spk_mat_all_shuf', ...
    'mean_rate_ordering', 'rate_mat_burst_only_rand', 'rate_mat_rand', ...
    'spk_times_rand', 'spk_times_id_rand', 'all_btb_corr_scores_av_shuf', ...
    'all_btb_corr_scores_std_shuf', 'av_btb_corr_scores_av_shuf', 'av_btb_corr_scores_std_shuf', ...
    'all_pw_corr_vals_av_shuf', 'all_pw_corr_vals_std_shuf', 'all_pw_corr_lags_av_shuf', ...
    'all_pw_corr_lags_std_shuf', 'burst_window', 'scaf_window', 'mean_cos_sim_av_shuf', ...
    'mean_cos_sim_std_shuf', 'cos_sim_av_shuf', 'cos_sim_std_shuf', 'mean_cos_sim_scaf', ...
    'mean_cos_sim_non_scaf', 'centered_pop_rate', 'sbsc_all_shuf', 'vars_all_shuf', ...
    'contributions_all_shuf', '-v7.3')


'All results saved'


%% functions

function t_spk_mat = remove_spk_outside_burst(t_spk_mat, edges)

    % for each burst
    for burst = 1:size(edges,1)
        
        % if this is the first burst
        if burst == 1
            
            % remove all spikes before burst
            t_spk_mat(1:edges(burst,1)-1,:,:)=0;
            
        else
            
            % remove all spikes since last burst
            t_spk_mat(edges(burst-1,2)+1 : edges(burst,1)-1, :, :)=0;
            
        end % if
        
        % if this is the last burst
        if burst == size(edges,1)
            
            % also remove all spikes after burst
            t_spk_mat(edges(burst,2)+1 : end, :, :)=0;
            
        end % if
        
    end % burst
    
end % fun remove_spk_outside_burst
    
% % %

function [rate_mat, spk_times, spk_times_id] = compute_rate_isi(t_spk_mat, GAUSS_SIGMA, TIME_RANGE)

    % make emtpy result arrays
    rate_mat = zeros(TIME_RANGE(2)*1000, size(t_spk_mat,2), size(t_spk_mat,3));
    spk_times = cell(1,size(t_spk_mat,2), size(t_spk_mat,3));
    spk_times_id = cell(1,size(t_spk_mat,2), size(t_spk_mat,3));
    
    % for each copy
    for c = 1:size(t_spk_mat,3)
    
        % for each unit
        for unit = 1:size(t_spk_mat,2)

            % %  spike times

            % obtain spike times in ms
            spk_times{unit, c} = find(t_spk_mat(:,unit, c));

            % remove spikes outside of time range under consideration
            spk_times{unit, c}(spk_times{unit, c} > TIME_RANGE(2)*1000) = [];
            spk_times{unit, c}(spk_times{unit, c} < TIME_RANGE(1)*1000) = [];

            % remove spike times below 0
            spk_times{unit, c}(spk_times{unit, c} <= 0) = [];

            % store spike time ids
            spk_times_id{unit, c} = unit*ones(1,length(spk_times{unit, c}));


            % %  firing rates

            % compute inter spike interval
            isi = diff(spk_times{unit, c});

            % ad spacer so that indices are the same
            isi = vertcat(NaN, isi);

            % compute firing rate based on isi
            isi_rate = 1./isi;

            % make temporary result array
            isi_rate_temp_result = zeros(1, TIME_RANGE(2)*1000);

            % for each spike except the first
            for spk = 2:length(spk_times{unit, c})

                % store rates at corresponding spike times
                isi_rate_temp_result(spk_times{unit, c}(spk-1):spk_times{unit, c}(spk)) = isi_rate(spk);

            end % spk

            % compute firing rates and store
            rate_mat(:,unit,c) = 1000*smoothdata(isi_rate_temp_result,'gaussian',GAUSS_SIGMA);

        end % unit
    end % c
    
end % fun compute_rate_isi

% % %

function [act_times, av_rate, cut_spk_mat] = compute_sig_per_burst(rate_mat, ...
    t_spk_mat, edges, tburst, PEAK_THRESH, MIN_SPIKES, CUT_RANGE)

% make empty result matrices and arrays
act_times = NaN(size(edges,1), size(rate_mat,2), size(rate_mat,3));
cut_rates = NaN(size(rate_mat,2), 1+sum(CUT_RANGE), size(edges,1), size(rate_mat,3));
cut_spk_mat = cell(size(rate_mat,2), size(rate_mat,3));

% for each burst
for burst = 1:size(edges,1) 
    
    % define relative time of burst peak to burst start
    t_burst_rel = tburst(burst) - edges(burst,1); 
    
    % for each copy
    for c = 1:size(rate_mat,3)
    
        % for each unit
        for unit = 1:size(rate_mat,2) 

            % intiate matrix if it is the first burst
            if burst == 1
                cut_spk_mat{unit, c} = zeros(size(edges,1), 1+sum(CUT_RANGE));
            end % if

            % obtain index of peak in burst range
            [peak_val,max_i] = max(rate_mat(edges(burst,1):edges(burst,2), unit, c));

            % if a rate peak is detected and population peak falls within the burst window
            if ~isempty(max_i) && peak_val > PEAK_THRESH || sum(rate_mat(edges(burst,1):edges(burst,2), unit, c)) >= MIN_SPIKES

                % save rate peak time with respect to the relative burst peak
                act_times(burst, unit, c) = max_i(1)-t_burst_rel; 

                % cut out firing rate relative to burst peak
                cut_rates(unit,:,burst, c) = rate_mat(tburst(burst)-CUT_RANGE(1):tburst(burst)+CUT_RANGE(2), unit, c)';

            end % if


            % cut out spike train relative to burst peak
            cut_spk_mat{unit, c}(burst,:) = t_spk_mat(tburst(burst)-CUT_RANGE(1):tburst(burst)+CUT_RANGE(2), unit, c);

        end % unit
    end % c
end % burst

% compute average rate from summed rate
av_rate = squeeze(mean(cut_rates,3,'omitnan'));

end % fun compute_sig_per_burst

% % %

function [all_burst_corr_scores, av_burst_corr_scores] = compute_burst_to_burst_corr(rate_mat, ...
    t_spk_mat, tburst, MAXLAG, WINDOW, MIN_SPIKES, MIN_FRAC)
 
% make empty result matrices
av_burst_corr_scores = NaN(1,size(rate_mat,2), size(rate_mat,3));
all_burst_corr_scores = NaN(size(rate_mat,2), length(tburst),length(tburst),size(rate_mat,3));

% for each copy
for c = 1:size(rate_mat,3)
    
    % for each unit
    for unit = 1:size(rate_mat,2)

        % make list of comparison bursts
        comp_bursts = 1:length(tburst);

        % set counter to 0
        counter = 0;

        % for each reference burst
        for ref_b = 1:length(tburst)

            % remove ref burst from comp bursts
            comp_bursts(comp_bursts == ref_b) = [];

            % count spikes for burst
            num_spikes_ref_b = sum(t_spk_mat(tburst(ref_b)-WINDOW(1):tburst(ref_b)+WINDOW(2), unit, c));

            % skip ref burst if there are less than MIN_SPIKES spikes
            if num_spikes_ref_b < MIN_SPIKES

                % add 1 to counter
                counter = counter + 1;

                % skip burst
                continue

            end % if

            % cut out firing rate for burst
            ref_rate = rate_mat(tburst(ref_b)-WINDOW(1):tburst(ref_b)+WINDOW(2), unit, c);

            % for each comparison burst
            for comp_b = 1:length(comp_bursts)

                % count spikes for burst
                num_spikes_comp_b = sum(t_spk_mat(tburst(comp_bursts(comp_b))-WINDOW(1):tburst(comp_bursts(comp_b))+WINDOW(2), unit, c));

                % skip ref burst if there are less than MIN_SPIKES spikes
                if num_spikes_comp_b < MIN_SPIKES
                    continue
                end % if

                % cut out firing rate for burst
                comp_rate = rate_mat(tburst(comp_bursts(comp_b))-WINDOW(1):tburst(comp_bursts(comp_b))+WINDOW(2), unit, c);

                % compute cross correlation
                [r, ~] = xcorr(ref_rate, comp_rate, MAXLAG, 'coeff');

                % obtain maximum correlation
                [max_corr, ~] = max(r);

                % store results
                all_burst_corr_scores(unit, comp_bursts(comp_b), ref_b, c) = max_corr;

            end % comp_b
        end % ref_b

        % if more than MIN_FRAC bursts had at least 2 spikes
        if counter/length(tburst) <= MIN_FRAC

            % select pairs
            to_average = all_burst_corr_scores(unit,:,:, c);
            
            % average results over all pairs
            av_burst_corr_scores(1, unit, c) = mean(to_average(:), 'omitnan');

        end % if

    end % unit
end % c

end % fun compute_burst_to_burst_corr

% % %

function [x_corr_vals, x_corr_lags] = compute_pairwise_corr(rate_mat, MAXLAG)

% make empty result arrays
x_corr_vals = zeros(size(rate_mat,2), size(rate_mat,2), size(rate_mat,3));
x_corr_lags = zeros(size(rate_mat,2), size(rate_mat,2), size(rate_mat,3));

% for each copy
for c = 1:size(rate_mat,3)
    
    sprintf('Computing correlations for dataset %.0f with %.0f units', c, size(rate_mat,2))
    
    % for each row unit
    for ref_unit = 1:size(rate_mat,2)

%         sprintf('Computing correlations for unit %.0f of %.0f', ref_unit, size(rate_mat,2))

        % for each column unit
        for comp_unit = 1:size(rate_mat,2)
            
            if ref_unit > comp_unit

                % compute cross correlation
                [corr_r, corr_lags] = xcorr(rate_mat(:,comp_unit,c), rate_mat(:,ref_unit,c), MAXLAG, 'coeff');

                % obtain maximum correlation
                [max_corr, max_corr_i] = max(corr_r);
                corr_opt_lag = corr_lags(max_corr_i);

                % store results in matrices
                x_corr_vals(comp_unit, ref_unit, c) = max_corr;
                x_corr_lags(comp_unit, ref_unit, c) = corr_opt_lag;
                x_corr_vals(ref_unit, comp_unit, c) = max_corr;
                x_corr_lags(ref_unit, comp_unit, c) = -corr_opt_lag;
                
            end % if

        end % unit_c

    end % unit_r

end % c

% remove NaN values
x_corr_vals(isnan(x_corr_vals)) = 0;

% remove lags of 0 corr values
x_corr_lags(x_corr_vals == 0) = NaN;


end % fun compute_pairwise_corr

% % %

function [mean_cos_sim, cos_sim, centered_pop_rate] = comp_burst_sim(rate_mat, pop_rate, frame_range, tburst)

% make empty result arrays
mean_cos_sim = zeros(length(frame_range),size(rate_mat,3));
cos_sim = zeros(length(tburst),length(tburst),length(frame_range),size(rate_mat,3));
centered_pop_rate = zeros(length(tburst),length(frame_range));
    
% for each copy
for c = 1:size(rate_mat,3)
    
    % normalize rate mat
    z_rate_mat = zscore(rate_mat(:,:,c));

    % for each frame
    for frame = 1:length(frame_range)

        % make empty result matrix
        frame_rates = NaN(length(tburst), size(z_rate_mat,2));

        % for each burst
        for burst = 1:length(tburst) 

            % obtain rate for frame and store
            frame_rates(burst, :) = z_rate_mat(int32(tburst(burst)+frame_range(frame)), :);

            % for each comparison burst
            for comp_burst = 1:length(tburst) 

                % obtain rate for frame of comparison burst
                frame_rate_comp = z_rate_mat(int32(tburst(comp_burst)+frame_range(frame)), :); 

                % compute cosine similarity between the two vectors and store
                cos_sim(burst,comp_burst,frame, c) = dot(frame_rates(burst, :), ...
                    frame_rate_comp)/(norm(frame_rates(burst, :))*norm(frame_rate_comp));

            end % comp_burst

            % if this is the first frame
            if frame == 1 && c == 1

                % store population rate relative to burst
                centered_pop_rate(burst,:) = pop_rate(int32(tburst(burst)+frame_range(1)) : int32(tburst(burst)+frame_range(end)));

            end % if

        end % burst

        % compute average cosine similarity
        to_average = cos_sim(:,:,frame,c);
        
%         ver = version;
%         year = regexp(ver, 'R(\d{4})', 'match');
%         if str2double(year{1}(2:end)) > 2017
        mean_cos_sim(frame, c) = mean(to_average(:), 'omitnan');

            

    end % frame
end % c

end % comp_burst_sim

% % %

function [sbsc, vars, contributions] = compute_manifolds(rate_mat, scafs, nscafs, norm_methd)

sbsc = cell(1,size(rate_mat, 3));
vars = cell(1,size(rate_mat, 3));
contributions = cell(1,size(rate_mat, 3));

% for each copy
for c = 1:size(rate_mat, 3)
    
    % seperating scaff/nscaff units
    X1 = rate_mat(:, :, c);
    X2 = rate_mat(:, scafs, c);
    X3 = rate_mat(:, nscafs, c);

    % mean center
    X1_ = X1 - mean(X1, 1);
    X2_ = X2 - mean(X2, 1);
    X3_ = X3 - mean(X3, 1);

    % normalize
    X1_ = normalize(X1_, 1, norm_methd);
    X2_ = normalize(X2_, 1, norm_methd);
    X3_ = normalize(X3_, 1, norm_methd);

    % set any zero divides to zero (some units might never spike)
    X1_(isnan(X1_)) = 0;
    X2_(isnan(X2_)) = 0;
    X3_(isnan(X3_)) = 0;

    % PCA
    [U1, S1, V1] = svd(X1_, 'econ');
    [U2, S2, V2] = svd(X2_, 'econ');
    [U3, S3, V3] = svd(X3_, 'econ');

    % Project spike dynamics onto PC axes
    sbsc1 = X1 * V1;
    sbsc2 = X2 * V2;
    sbsc3 = X3 * V3;

    % Calculate squared singular values
    S1_squared = S1.^2;
    S2_squared = S2.^2;
    S3_squared = S3.^2;
    
    % Explained variance from singular values
    var1 = diag(S1_squared) ./ sum(diag(S1_squared));
    var2 = diag(S2_squared) ./ sum(diag(S2_squared));
    var3 = diag(S3_squared) ./ sum(diag(S3_squared));
    var1 = (var1 ./ sum(var1)) .* 100;
    var2 = (var2 ./ sum(var2)) .* 100;
    var3 = (var3 ./ sum(var3)) .* 100;
    

    
    
    % Normalize squared singular values to get contribution ratio
    total_squared1 = sum(S1_squared);
    contribution_ratio1 = S1_squared / total_squared1;
    total_squared2 = sum(S2_squared);
    contribution_ratio2 = S2_squared / total_squared2;
    total_squared3 = sum(S3_squared);
    contribution_ratio3 = S3_squared / total_squared3;
    
    [~, num_features1] = size(U1);
    num_components1 = length(S1);
    [~, num_features2] = size(U2);
    num_components2 = length(S2);
    [~, num_features3] = size(U3);
    num_components3 = length(S3);
    
    contributions1 = zeros(num_features1, num_components1);
    
    % Calculate contributions of each variable to each component
    for i = 1:num_components1
        component_contributions = (V1(:, i).^2) .* contribution_ratio1(i);
        contributions1(:, i) = component_contributions / sum(component_contributions);
    end

    contributions2 = zeros(num_features2, num_components2);
    
    % Calculate contributions of each variable to each component
    for i = 1:num_components2 
        component_contributions = (V2(:, i).^2) .* contribution_ratio2(i);
        contributions2(:, i) = component_contributions / sum(component_contributions);
    end
    
    contributions3 = zeros(num_features3, num_components3);
    
    % Calculate contributions of each variable to each component
    for i = 1:num_components3
        component_contributions = (V3(:, i).^2) .* contribution_ratio3(i);
        contributions3(:, i) = component_contributions / sum(component_contributions);
    end

    
    % output the subspace axes and respective variances
    sbsc{c} = struct(); % subspaces
    sbsc{c}.all    = sbsc1; 
    sbsc{c}.scaff  = sbsc2; 
    sbsc{c}.nscaff = sbsc3;
    vars{c} = struct(); % variances
    vars{c}.all    = var1; 
    vars{c}.scaff  = var2; 
    vars{c}.nscaff = var3; 
    contributions{c} = struct(); % contributions
    contributions{c}.all = contributions1;
    contributions{c}.scaff = contributions2;
    contributions{c}.nscaff = contributions3;
    
end % c

end

% % %

function peak_rel = compute_trel_brst_peak(tburst, burst_window, n_snapshots)
    peak_rel = ones(n_snapshots, 1)*615;
    burst_range = burst_window(1):burst_window(end);
    % Iterate through each burst peak
    for i = 1:length(tburst)
        % Set the values around each peak to time relative to burst peak 
        peak_rel(tburst(i)+burst_window(1):tburst(i)+burst_window(end)) = burst_range;
    end
end