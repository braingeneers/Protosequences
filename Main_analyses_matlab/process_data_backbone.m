%% Load data

% load data
load('t_spk_mat_sorted.mat')

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

% smooth summed spike times with square window
square_smooth_summed_spike = smoothdata(sum(t_spk_mat, 2),'movmean',SQUARE_WIDTH);

% smooth smoothed spike times with gaussian window
pop_rate = smoothdata(square_smooth_summed_spike,'gaussian',GAUSS_SIGMA);

% compute rms of pop rate
pop_rms = rms(pop_rate);

% detect peaks
[peak_amp, peaks] = findpeaks(pop_rate, "MinPeakHeight", pop_rms*THR_BURST, "MinPeakDistance", MIN_BURST_DIFF);

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

if TIME_RANGE(2)*1000-tburst(end)<=1000
    tburst(end) = [];
    edges(end,:) = [];
end


%% compute scaffold units

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
GAUSS_SIGMA = 50; % sigma for gaussian smoothing in ms
SHUFF_PER_SPK = 5; % number of shuffling operations per spike

seed = 1;

% set seed
rng(seed)

% obtain t_spk_mat for bursts only
t_spk_mat_burst_only = remove_spk_outside_burst(t_spk_mat, edges);

% randomize spike matrices
t_spk_mat_rand = randomize_spk_mat(t_spk_mat, SHUFF_PER_SPK);
t_spk_mat_burst_only_rand = randomize_spk_mat(t_spk_mat_burst_only, SHUFF_PER_SPK);

% count number of spikes per unit
spk_count = sum(t_spk_mat);

% compute firing rate per unit
[rate_mat, spk_times, spk_times_id] = compute_rate_isi(t_spk_mat, GAUSS_SIGMA, TIME_RANGE);
[rate_mat_burst_only, ~, ~] = compute_rate_isi(t_spk_mat_burst_only, GAUSS_SIGMA, TIME_RANGE);
[rate_mat_rand, spk_times_rand, spk_times_id_rand] = compute_rate_isi(t_spk_mat_rand, GAUSS_SIGMA, TIME_RANGE);
[rate_mat_burst_only_rand, spk_times_burst_only_rand, spk_times_id_burst_only_rand] = compute_rate_isi(t_spk_mat_burst_only_rand, GAUSS_SIGMA, TIME_RANGE);


%% compute firing rate peaks and average signals per burst
 
% specify plot parameters
PEAK_THRESH = 1000; % minimum peak firing rate
MIN_SPIKES = 2;
CUT_RANGE = [250,500]; % time before and after burst peak to plot (ms)

% compute burst signal metrics per burst
[act_times, av_rate, cut_spk_mat] = compute_sig_per_burst(rate_mat, ...
    t_spk_mat, edges, tburst, PEAK_THRESH, MIN_SPIKES, CUT_RANGE);
[act_times_rand, av_rate_rand, cut_spk_mat_rand] = compute_sig_per_burst(rate_mat_burst_only_rand, ...
    t_spk_mat_burst_only_rand, edges, tburst, PEAK_THRESH, MIN_SPIKES, CUT_RANGE);


%%  compute reordering of units based on scaf and median act time

% set sort metric to median of act times for bursts with at least 2 spikes
act_times_copy = act_times;
act_times_copy(above_thresh==0) = NaN;
sort_metric = median(act_times_copy, "omitnan");

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
    t_spk_mat, tburst, MAXLAG_BTB, WINDOW, MIN_SPIKES, MIN_BURST_FRAC);
 
% compute burst to burst correlations for randomized data
[all_btb_corr_scores_rand, av_btb_corr_scores_rand] = compute_burst_to_burst_corr(rate_mat_burst_only_rand, ...
    t_spk_mat_burst_only_rand, tburst, MAXLAG_BTB, WINDOW, MIN_SPIKES, MIN_BURST_FRAC);
 

%% compute pairwise cross correlation scores

% specify max lag
MAXLAG_PW = 50;

% compute pairwise cross correlations for normal data
[all_pw_corr_vals, all_pw_corr_lags] = compute_pairwise_corr(rate_mat_burst_only, MAXLAG_PW);

% compute pairwise cross correlations for randomized data
[all_pw_corr_vals_rand, all_pw_corr_lags_rand] = compute_pairwise_corr(rate_mat_burst_only_rand, MAXLAG_PW);


%% compute burst similarity score

% specify time relative to burst peak to analyze
FRAME_RANGE = [-450:1000];

[mean_cos_sim, cos_sim, centered_pop_rate] = comp_burst_sim(rate_mat, pop_rate, FRAME_RANGE, tburst);
[mean_cos_sim_rand, cos_sim_rand, ~] = comp_burst_sim(rate_mat_burst_only_rand, pop_rate, FRAME_RANGE, tburst);
[mean_cos_sim_scaf, cos_sim_scaf, ~] = comp_burst_sim(rate_mat(:,scaf_units), pop_rate, FRAME_RANGE, tburst);
[mean_cos_sim_non_scaf, cos_sim_non_scaf, ~] = comp_burst_sim(rate_mat(:,non_scaf_units), pop_rate, FRAME_RANGE, tburst);


%% perform PCA analysis

norm_methd = "scale";

% compute manifolds
[sbsc, vars, contributions] = compute_manifolds(rate_mat_burst_only, scaf_units, non_scaf_units, norm_methd);
[sbsc_rand, vars_rand, contributions_rand] = compute_manifolds(rate_mat_burst_only_rand, scaf_units, non_scaf_units, norm_methd);

% compute times relative to closest burst peak per frame (for plotting)
peak_rel = compute_trel_brst_peak(tburst, burst_window, (TIME_RANGE(2)-TIME_RANGE(1))*1000);


%% save results

% save results for normal data
save("single_recording_metrics", "edges", "tburst", "pop_rate", "spk_count", "t_spk_mat", ... 
    "above_thresh", "frac_per_unit", "frac_per_burst", "scaf_units", ...
    "non_scaf_units", "act_times", "av_rate", "cut_spk_mat", ...
    "mean_rate_ordering", "rate_mat", "rate_mat_burst_only", "spk_times", ...
    "spk_times_id", "all_btb_corr_scores", "av_btb_corr_scores", ...
    "all_pw_corr_vals", "all_pw_corr_lags", "burst_window", "scaf_window", ...
    "mean_cos_sim", "cos_sim", "mean_cos_sim_scaf", "cos_sim_scaf", "mean_cos_sim_non_scaf", ...
    "cos_sim_non_scaf", "centered_pop_rate", "sbsc", "vars", "contributions", ...
    "-v7.3")

% save results for randomized data
save("single_recording_metrics_rand", "edges", "tburst", "pop_rate", "spk_count", "t_spk_mat_rand", ... 
    "above_thresh", "frac_per_unit", "frac_per_burst", "scaf_units", ...
    "non_scaf_units", "act_times_rand", "av_rate_rand", "cut_spk_mat_rand", ...
    "mean_rate_ordering", "rate_mat_burst_only_rand", "rate_mat_rand", ...
    "all_btb_corr_scores_rand", "spk_times_rand", "spk_times_id_rand", ...
    "av_btb_corr_scores_rand", "all_pw_corr_vals_rand", "all_pw_corr_lags_rand", ...
    "burst_window", "scaf_window", "mean_cos_sim_rand", "cos_sim_rand", "mean_cos_sim_scaf", ...
    "mean_cos_sim_non_scaf", "centered_pop_rate", "sbsc_rand", "vars_rand", "contributions_rand", ...
    "-v7.3")

"All results saved"


%% functions

function t_spk_mat = remove_spk_outside_burst(t_spk_mat, edges)

    % for each burst
    for burst = 1:size(edges,1)
        
        % if this is the first burst
        if burst == 1
            
            % remove all spikes before burst
            t_spk_mat(1:edges(burst,1)-1,:)=0;
            
        else
            
            % remove all spikes since last burst
            t_spk_mat(edges(burst-1,2)+1 : edges(burst,1)-1, :)=0;
            
        end % if
        
        % if this is the last burst
        if burst == size(edges,1)
            
            % also remove all spikes after burst
            t_spk_mat(edges(burst,2)+1 : end, :)=0;
            
        end % if
        
    end % burst
    
end % fun remove_spk_outside_burst
    
% % %

function [rate_mat, spk_times, spk_times_id] = compute_rate_isi(t_spk_mat, GAUSS_SIGMA, TIME_RANGE)

    % make emtpy result arrays
    rate_mat = zeros(TIME_RANGE(2)*1000, size(t_spk_mat,2));
    spk_times = cell(1,size(t_spk_mat,2));
    spk_times_id = cell(1,size(t_spk_mat,2));
    
    % for each unit
    for unit = 1:size(t_spk_mat,2)
    
        % %  spike times
        
        % obtain spike times in ms
        spk_times{unit} = find(t_spk_mat(:,unit));

        % remove spikes outside of time range under consideration
        spk_times{unit}(spk_times{unit} > TIME_RANGE(2)*1000) = [];
        spk_times{unit}(spk_times{unit} < TIME_RANGE(1)*1000) = [];
        
        % remove spike times below 0
        spk_times{unit}(spk_times{unit} <= 0) = [];

        % store spike time ids
        spk_times_id{unit} = unit*ones(1,length(spk_times{unit}));
        
        
        % %  firing rates
        
        % compute inter spike interval
        isi = diff(spk_times{unit});

        % ad spacer so that indices are the same
        isi = vertcat(NaN, isi);

        % compute firing rate based on isi
        isi_rate = 1./isi;

        % make temporary result array
        isi_rate_temp_result = zeros(1, TIME_RANGE(2)*1000);

        % for each spike except the first
        for spk = 2:length(spk_times{unit})

            % store rates at corresponding spike times
            isi_rate_temp_result(spk_times{unit}(spk-1):spk_times{unit}(spk)) = isi_rate(spk);

        end % spk

        % compute firing rates and store
        rate_mat(:,unit) = 1000*smoothdata(isi_rate_temp_result,'gaussian',GAUSS_SIGMA);

    end % unit
    
end % fun compute_rate_isi

% % %

function rand_spk_mat = randomize_spk_mat(spk_mat, rand_per_spk)

% copy spike mat
rand_spk_mat = spk_mat;

% set nan output to false
NAN_OUT = false;

% for each shuffling operations
for s_o = 1:(sum(spk_mat, "all") * rand_per_spk)
    
    if mod(s_o-1,1000) == 0
        sprintf("%.0f of %.0f", s_o, sum(spk_mat, "all") * rand_per_spk)
    end
    
    % get all linear index values of spikes in spk_mat
    spk_lin_i = find(rand_spk_mat == 1);

    % set SEARCH_PAIR to True
    SEARCH_PAIR = true;
    
    % set counter to 0
    counter = 0;
    
    % while pair should be searched
    while SEARCH_PAIR == true && counter < 1000
        
        % randomly select two frames with a spike
        pair_selection = randi(length(spk_lin_i),2);
        
        % obtain row and column values of selected spikes
        [row1,col1] = ind2sub(size(rand_spk_mat),spk_lin_i(pair_selection(1)));
        [row2,col2] = ind2sub(size(rand_spk_mat),spk_lin_i(pair_selection(2)));
    
        % if the selected spikes don't fall in same frame or come from same unit
        if row1~=row2 && col1~=col2 && rand_spk_mat(row1,col2)==0 && rand_spk_mat(row2,col1)==0
            
            % set SEARCH_PAIR to false
            SEARCH_PAIR = false;
            
        else
            
            % add 1 to counter
            counter = counter + 1;
            
        end % if
        
    end % while
    
    % if while loop was escaped due to counter = 1000
    if counter >= 1000
        
        % print warning message
        print("WARNING: eligable spike pair could not be found")
    
    else
        
        % switch spikes
        rand_spk_mat(row1,col1) = 0;
        rand_spk_mat(row2,col2) = 0;
        rand_spk_mat(row1,col2) = 1;
        rand_spk_mat(row2,col1) = 1;
        
    end % if
       
end % s_o


% % test randomization results

% test if rand_spk_mat has same number of spikes
if sum(sum(rand_spk_mat)) ~= sum(sum(spk_mat))
    "ERROR: Randomized spike matrix has different number of spikes"
    NAN_OUT = true;
end % if

% test if every unit has same average rate
if sum(sum(rand_spk_mat) == sum(spk_mat)) ~= size(rand_spk_mat,2)
    "ERROR: Average rate per unit is different"
    NAN_OUT = true;
end % if

% test if population rate is the same
if sum(sum(rand_spk_mat,2) == sum(spk_mat,2)) ~= size(rand_spk_mat,1)
    "ERROR: Population rate is different"
    NAN_OUT = true;
end % if

if NAN_OUT == true
    rand_spk_mat = NaN;
end

end % fun randomize_spk_mat

% % %

function [act_times, av_rate, cut_spk_mat] = compute_sig_per_burst(rate_mat, ...
    t_spk_mat, edges, tburst, PEAK_THRESH, MIN_SPIKES, CUT_RANGE)

% make empty result matrices and arrays
act_times = NaN(size(edges,1), size(rate_mat,2));
cut_rates = NaN(size(rate_mat,2), 1+sum(CUT_RANGE), size(edges,1));
cut_spk_mat = cell(1,size(rate_mat,2));

% for each burst
for burst = 1:size(edges,1) 
    
    % define relative time of burst peak to burst start
    t_burst_rel = tburst(burst) - edges(burst,1); 
    
    % for each unit
    for unit = 1:size(rate_mat,2) 
        
        % intiate matrix if it is the first burst
        if burst == 1
            cut_spk_mat{unit} = zeros(size(edges,1), 1+sum(CUT_RANGE));
        end % if
        
        % obtain index of peak in burst range
        [peak_val,max_i] = max(rate_mat(edges(burst,1):edges(burst,2), unit));
        
        % if a rate peak is detected and population peak falls within the burst window
        if ~isempty(max_i) && peak_val > PEAK_THRESH || sum(rate_mat(edges(burst,1):edges(burst,2), unit)) >= MIN_SPIKES
            
            % save rate peak time with respect to the relative burst peak
            act_times(burst, unit) = max_i(1)-t_burst_rel; 
            
            % cut out firing rate relative to burst peak
            cut_rates(unit,:,burst) = rate_mat(tburst(burst)-CUT_RANGE(1):tburst(burst)+CUT_RANGE(2), unit)';
        
        end % if
        
        
        % cut out spike train relative to burst peak
        cut_spk_mat{unit}(burst,:) = t_spk_mat(tburst(burst)-CUT_RANGE(1):tburst(burst)+CUT_RANGE(2), unit);
        
    end % unit
    
end % burst

% compute average rate from summed rate
av_rate = mean(cut_rates,3,"omitnan");

end % fun compute_sig_per_burst

% % %

function [all_burst_corr_scores, av_burst_corr_scores] = compute_burst_to_burst_corr(rate_mat, ...
    t_spk_mat, tburst, MAXLAG, WINDOW, MIN_SPIKES, MIN_FRAC)
 
 % make empty result matrices
 av_burst_corr_scores = NaN(1,size(rate_mat,2));
 all_burst_corr_scores = NaN(size(rate_mat,2), length(tburst),length(tburst));
    
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
        num_spikes_ref_b = sum(t_spk_mat(tburst(ref_b)-WINDOW(1):tburst(ref_b)+WINDOW(2), unit));
        
        % skip ref burst if there are less than MIN_SPIKES spikes
        if num_spikes_ref_b < MIN_SPIKES
            
            % add 1 to counter
            counter = counter + 1;
            
            % skip burst
            continue
            
        end % if
        
        % cut out firing rate for burst
        ref_rate = rate_mat(tburst(ref_b)-WINDOW(1):tburst(ref_b)+WINDOW(2), unit);
        
        % for each comparison burst
        for comp_b = 1:length(comp_bursts)
                        
            % count spikes for burst
            num_spikes_comp_b = sum(t_spk_mat(tburst(comp_bursts(comp_b))-WINDOW(1):tburst(comp_bursts(comp_b))+WINDOW(2), unit));

            % skip ref burst if there are less than MIN_SPIKES spikes
            if num_spikes_comp_b < MIN_SPIKES
                continue
            end % if
        
            % cut out firing rate for burst
            comp_rate = rate_mat(tburst(comp_bursts(comp_b))-WINDOW(1):tburst(comp_bursts(comp_b))+WINDOW(2), unit);
            
            % compute cross correlation
            [r, ~] = xcorr(ref_rate, comp_rate, MAXLAG, 'coeff');
            
            % obtain maximum correlation
            [max_corr, ~] = max(r);
                                    
            % store results
            all_burst_corr_scores(unit, comp_bursts(comp_b), ref_b) = max_corr;
        
        end % comp_b
    end % ref_b
    
    % if more than MIN_FRAC bursts had at least 2 spikes
    if counter/length(tburst) <= MIN_FRAC
    
        % average results over all pairs
        av_burst_corr_scores(unit) = mean(all_burst_corr_scores(unit,:,:), "all", "omitnan");
    
    end % if
    
end % unit

end % fun compute_burst_to_burst_corr

% % %

function [x_corr_vals, x_corr_lags] = compute_pairwise_corr(rate_mat, MAXLAG)

% make empty result arrays
x_corr_vals = zeros(size(rate_mat,2), size(rate_mat,2));
x_corr_lags = zeros(size(rate_mat,2), size(rate_mat,2));

% for each row unit
for ref_unit = 1:size(rate_mat,2)
    
    sprintf("Computing correlations for unit %.0f of %.0f", ref_unit, size(rate_mat,2))
    
    % for each column unit
    for comp_unit = 1:size(rate_mat,2)
            
        % compute cross correlation
        [corr_r, corr_lags] = xcorr(rate_mat(:,comp_unit), rate_mat(:,ref_unit), MAXLAG, 'coeff');
        
        % obtain maximum correlation
        [max_corr, max_corr_i] = max(corr_r);
        corr_opt_lag = corr_lags(max_corr_i);

        % store results in matrices
        x_corr_vals(comp_unit, ref_unit) = max_corr;
        x_corr_lags(comp_unit, ref_unit) = corr_opt_lag;
        
    end % unit_c
    
end % unit_r

% remove NaN values
x_corr_vals(isnan(x_corr_vals)) = 0;

% remove lags of 0 corr values
x_corr_lags(x_corr_vals == 0) = 0;

end % fun compute_pairwise_corr

% % %

function [mean_cos_sim, cos_sim, centered_pop_rate] = comp_burst_sim(rate_mat, pop_rate, frame_range, tburst)

% normalize rate mat
z_rate_mat = zscore(rate_mat);

% make empty result arrays
mean_cos_sim = zeros(1,length(frame_range));
cos_sim = zeros(length(tburst),length(tburst),length(frame_range));
centered_pop_rate = zeros(length(tburst),length(frame_range));

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
            cos_sim(burst,comp_burst,frame) = dot(frame_rates(burst, :), ...
                frame_rate_comp)/(norm(frame_rates(burst, :))*norm(frame_rate_comp));
            
        end % comp_burst
        
        % if this is the first frame
        if frame == 1
            
            % store population rate relative to burst
            centered_pop_rate(burst,:) = pop_rate(int32(tburst(burst)+frame_range(1)) : int32(tburst(burst)+frame_range(end)));
            
        end % if
        
    end % burst

    % compute average cosine similarity
    mean_cos_sim(frame) = mean(cos_sim(:,:,frame), "all", "omitnan");
    
end % frame

end % comp_burst_sim

% % %

function [sbsc, vars, contributions] = compute_manifolds(rate_mat, scafs, nscafs, norm_methd)

    % seperating scaff/nscaff units
    X1 = rate_mat;
    X2 = rate_mat(:, scafs);
    X3 = rate_mat(:, nscafs);

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
    sbsc = struct(); % subspaces
    sbsc.all    = sbsc1; 
    sbsc.scaff  = sbsc2; 
    sbsc.nscaff = sbsc3;
    vars = struct(); % variances
    vars.all    = var1; 
    vars.scaff  = var2; 
    vars.nscaff = var3; 
    contributions = struct(); % contributions
    contributions.all = contributions1;
    contributions.scaff = contributions2;
    contributions.nscaff = contributions3;
    
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