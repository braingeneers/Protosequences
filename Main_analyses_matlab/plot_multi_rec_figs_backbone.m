%% load data

% % add relevant toolboxes to path
addpath(genpath("/home/vandermolen/matlab/firing_chains/final_scripts"))

% % specify which organoid to analyse
% rec_paths = [paths to folders with files]; % !!! adjust path and file names for loading data

% make empty cell arrays for results
edges = cell(1,length(rec_paths));
tburst = cell(1,length(rec_paths));
spk_count = cell(1,length(rec_paths));
above_thresh = cell(1,length(rec_paths));
frac_per_unit = cell(1,length(rec_paths));
frac_per_burst = cell(1,length(rec_paths));
scaf_units = cell(1,length(rec_paths));
non_scaf_units = cell(1,length(rec_paths));
rate_mat = cell(1,length(rec_paths));
act_times = cell(1,length(rec_paths));
av_btb_corr_scores = cell(1,length(rec_paths));
all_pw_corr_vals = cell(1,length(rec_paths));
burst_windows = cell(1,length(rec_paths));
scaf_windows = cell(1,length(rec_paths));
mean_cos_sim = cell(1,length(rec_paths));
mean_rate_ordering = cell(1,length(rec_paths));
var_vals = cell(1,length(rec_paths));

% for each array
for array = 1:length(rec_paths)
        
    % split file identifier
    split_name = split(rec_paths(array), "/");
    
    % % load data of first treatment
    
    % load single recording data for organoid
    s_rec_met = load(sprintf('%s/single_recording_metrics.mat', rec_paths(array)));
    
    % store results in cell array
    edges{array} = s_rec_met.edges;
    tburst{array} = s_rec_met.tburst;
    spk_count{array} = s_rec_met.spk_count;
    above_thresh{array} = s_rec_met.above_thresh;
    frac_per_unit{array} = s_rec_met.frac_per_unit;
    frac_per_burst{array} = s_rec_met.frac_per_burst;
    scaf_units{array} = s_rec_met.scaf_units;
    non_scaf_units{array} = s_rec_met.non_scaf_units;
    rate_mat{array} = s_rec_met.rate_mat;
    act_times{array} = s_rec_met.act_times;
    av_btb_corr_scores{array} = s_rec_met.av_btb_corr_scores;
    all_pw_corr_vals{array} = s_rec_met.all_pw_corr_vals;
    burst_windows{array} = s_rec_met.burst_window;
    scaf_windows{array} = s_rec_met.scaf_window;
    mean_cos_sim{array} = s_rec_met.mean_cos_sim;
    mean_rate_ordering{array} = s_rec_met.mean_rate_ordering;
    var_vals{array} = s_rec_met.vars;

end % array


%% set recording names
% rec_names = ["Or1", "Or2", "Or3", "Or4", "Or5", "Or6", "Or7", "Or8"];
% rec_names = ["M1S1", "M1S2", "M2S1", "M2S2", "M3S1", "M3S2"];
rec_names = ["Or1", "Or2", "Or3", "Or4", "Or5", "Or6", "Or7", "Or8", "M1S1", ...
    "M1S2", "M2S1", "M2S2", "M3S1", "M3S2", "Pr1", "Pr2", "Pr3", "Pr4", ...
    "Pr5", "Pr6", "Pr7", "Pr8", "Pr9", "Pr10"];


%% Show Log-Normal fit to firing rate distribution
% (Fig S1A-C)
% S1A-C = Load organoid recordings only

% select parameters
ARRAY_OI = 1; % which array to use for panel A
BIN_EDGES = linspace(0,10,21); % binning data for histogram
FIT_LN_RANGE = [-5,2]; % fitting range for ln of rate

% define bin centers
bin_centers = BIN_EDGES(2:end)-0.5*diff(BIN_EDGES);

% define log-normal distribution
log_norm_dist = @(b,x) 1./(x(:,1) .* b(1) .* sqrt(2 .* pi)) .* exp(-1 .* ( ( log(x(:,1)) - b(2) ).^2 / ( 2 .* b(1) ).^2 )) ;
est_initial_log_norm_dist = [1, 0.25];

% define normal distribution
norm_dist = @(b,x) 1/(b(1) * sqrt(2 * pi)) * exp(-0.5 * ( (x(:,1) - b(2) ) ./ b(1) ).^2);
est_initial_norm_dist = [1, 0.25];

% make empty result cell array for average firing rates
av_rate = cell(1,length(spk_count));

% for each array
for array = 1:length(spk_count)
    
    % compute average firing rate per unit
    av_rate{array} = spk_count{array}./(size(rate_mat{array},1)/1000);
    
end % array


% initiate figure
cm = figure(1);
clf

% adjust size of figure
set(gcf,'PaperPositionMode','auto')
set(cm, 'Position', [0 20 1300 300])
set(cm, 'Renderer', 'painters')

% initiate empty result arrays
r_squared_vals = zeros(1, size(av_rate, 2));

% for each array
for array = 1:size(av_rate, 2)
    
    % obtain hist counts
    counts = histcounts(av_rate{array}, BIN_EDGES,'Normalization','probability');
    
    % find maximum index with nonzero values
    max_index = max(find(counts > 0));
    
    % combine x and y values in table
    tbl = table(bin_centers(1:max_index)', counts(1:max_index)');
    
    % fit model
    mdl = fitnlm(tbl, log_norm_dist, est_initial_log_norm_dist);
    
    % Extract the coefficient values from the the model object.
    coefficients = mdl.Coefficients{:, 'Estimate'};
    
    % get x and y values for fitted model
    xFitted = linspace(min(bin_centers), max(bin_centers), 301)';
    yFitted = log_norm_dist(coefficients,xFitted);
    
    % obtain r_squared value of fit
    r_squared_vals(array) = mdl.Rsquared.Ordinary;
    
    % call subplot for fitted models
    subplot(1,3,2)
    hold on
    
    % make histogram of log(rate) values
    [log_counts, edges] = histcounts(log(av_rate{array}), ceil(length(log(av_rate{array}))^0.4), 'Normalization','probability');
    
    % compute bin centers for ln data
    bin_centers_ln_plot = edges(2:end)-diff(edges);
    
    % find maximum index with nonzero values
    max_index = max(find(log_counts > 0));
    
    % combine x and y values in table
    tbl = table(bin_centers_ln_plot(1:max_index)', log_counts(1:max_index)');
    
    % fit normal distribution
    mdl = fitnlm(tbl, norm_dist, est_initial_norm_dist);
    
    % Extract the coefficient values from the the model object.
    coefficients = mdl.Coefficients{:, 'Estimate'};
    
    % get x and y values for fitted model
    ln_xFitted = linspace(FIT_LN_RANGE(1), FIT_LN_RANGE(2), 301)';
    ln_yFitted = norm_dist(coefficients,ln_xFitted);
    
    % plot fitted model
    plot(ln_xFitted, ln_yFitted, "LineWidth", 2)
    
    % if this is the array of interest
    if array == ARRAY_OI
        
        % call subplot for example
        subplot(1,3,1)
        hold on
        
        % plot fitted model
        plot(xFitted, yFitted, "k--", "LineWidth", 1)
        
        % plot results
        scatter(bin_centers, counts, 20, "ko", "Filled", "MarkerFaceAlpha", 0.5)
        
    end % if
    
end % array


% % general figure layout
        
% improve layout for example distribution figure
subplot(1,3,1)

xlabel("Av. rate (Hz)")
ylabel("Fraction of units")
ax = gca;
ax.FontSize = 14;
set(gca,'linewidth',3)

% % add inset
axes('Position', [.22 .5 .1 .35]);  %[left bottom width height]
hold on

% bin ln data
[counts, bin_lin_edges] = histcounts(log(av_rate{ARRAY_OI}), ceil(length(log(av_rate{ARRAY_OI}))^0.4), 'Normalization','probability');

% compute bin centers for ln data
bin_centers_ln_plot = bin_lin_edges(2:end)-diff(bin_lin_edges);

% plot data points of binned data
scatter(bin_centers_ln_plot, counts, 20, "ko", "Filled", "MarkerFaceAlpha", 0.5)

% find maximum index with nonzero values
max_index = max(find(counts > 0));

% combine x and y values in table
tbl = table(bin_centers_ln_plot(1:max_index)', counts(1:max_index)');

% fit normal distribution
mdl = fitnlm(tbl, norm_dist, est_initial_norm_dist);
   
% Extract the coefficient values from the the model object.
coefficients = mdl.Coefficients{:, 'Estimate'};
        
% get x and y values for fitted model
xFitted = linspace(FIT_LN_RANGE(1), FIT_LN_RANGE(2), 301)';
yFitted = norm_dist(coefficients,xFitted);
        
% plot fitted model
plot(xFitted, yFitted, "k--", "LineWidth", 1)       
        
% adjust axes
xlabel("LN(Av. rate) (Hz)")
ylabel("Fraction of units")
ax = gca;
ax.FontSize = 14;
set(gca,'linewidth',3)


% improve layout for subplot with results per array
subplot(1,3,2)

% add axis labels
xlabel("LN(Av. rate) (Hz)")
ylabel("Fraction of units")

% add legend
legend(rec_names, "Location", "NorthEast", 'NumColumns', 2)
legend("boxoff")

% adjuste axes
ylim_curr = ylim;
ylim([ylim_curr(1), ylim_curr(2)*1.2])
xlim(FIT_LN_RANGE)
ax = gca;
ax.FontSize = 14;
set(gca,'linewidth',3)


% initiate subplot for r_squared values
subplot(1,3,3)

% plot results
bar(r_squared_vals)

xticks([1:size(av_rate, 2)])
xticklabels(rec_names)
ylabel("R² for log-normal fit")
ylim([0,1])

ax = gca;
ax.FontSize = 14;
set(gca,'linewidth',3)
box off


%% compare average firing rate
% (Fig S1D, S23)
% S1D = Load organoid recordings only
% S23 = Load all recordings

% only plot results for units with at least MIN_SPIKES in recording
MIN_SPIKES = 30;

% specify plot colors
box_colors = repmat(["r", "b"], 1,length(spk_count));

% make empty result arrays
vectorized_data = [];
unit_type_group = [];
org_type_group = [];
boxplot_group = [];
tick_locs = zeros(1,length(spk_count));

% for each array
for array = 1:length(spk_count)
    
    % compute average rate per unit
    av_rate_values = spk_count{array}/(size(rate_mat{array},1) / 1000);
    
    % set scores for units with less than MIN_SPIKES to NaN
    av_rate_values(spk_count{array} < MIN_SPIKES) = NaN;
    
    % store data
    vectorized_data = [vectorized_data, av_rate_values(scaf_units{array}), av_rate_values(non_scaf_units{array})];
    
    % store unit type labels
    unit_type_group = [unit_type_group, ones(1,length(scaf_units{array})), zeros(1,length(non_scaf_units{array}))];
    org_type_group = [org_type_group, array*ones(1,length(av_rate_values))];
    boxplot_group = [boxplot_group, (1+(array-1)*3)*ones(1,length(scaf_units{array})), (2+(array-1)*3)*ones(1,length(non_scaf_units{array}))];
    
    % store tick location
    tick_locs(array) = 1.5+(array-1)*2;        
        
end % array

% intiate figure
fig = figure(2);
clf

% adjust size of figure
set(gcf,'PaperPositionMode','auto')
set(fig, 'Position', [100 100 400 300])
set(fig, 'Renderer', 'painters')

% plot boxplot of consistency scores
boxplot(vectorized_data, boxplot_group);

% obtain individual boxes
bh = findobj(gca,'Tag','Box');

% color individual boxes
for j=1:length(bh)
    patch(get(bh(j),'XData'),get(bh(j),'YData'),box_colors(j),'FaceAlpha',.5);
end

% add y-axis label
ylabel("Av. firing rate (Hz)")
set(gca, 'YScale', 'log')

% adjust ticks
xticks(tick_locs)
xticklabels(rec_names)

set(gca,'linewidth',3)
set(gca, 'fontsize', 14)
set(gca, 'YScale', 'log')
box off


%% plot scaffold units
% (Fig S2A-B, S5A)
% S2A-B = Load all recordings
% S6A = Load Or2_8M and Or2_8M_4H from the side_experiments folder

% initiate figure
fig = figure(3);
clf

% make empty result arrays
frac_scaf = zeros(1,length(frac_per_unit));
bar_data = zeros(length(frac_per_unit), 2);
bar_error = zeros(length(frac_per_unit), 2);

% adjust size of figure
set(fig,'PaperPositionMode','auto')
set(fig, 'Position', [100 100 1400 700])
set(fig, 'Renderer', 'painters')

% make subplot with fraction of bursts with at least MIN_SPIKES spikes
subplot(2,1,1)
hold on

% plot results as violinplot
[~, lh, ~, ~, ~] = violin(frac_per_unit, 'facecolor', [1,1,1], 'medc', [], 'mc', [], 'bw', 0.1);

% for each array
for array = 1:length(frac_per_unit)
    
    % compute fraction of scaffold units
    frac_scaf(array) = length(scaf_units{array})/length(frac_per_unit{array});
    
    % store data for grouped barplot
    bar_data(array, 2) = mean(frac_per_burst{array}, "omitnan")-length(scaf_units{array})/length(frac_per_unit{array});
    bar_data(array, 1) = length(scaf_units{array})/length(frac_per_unit{array});
    bar_error(array, 2) = std(frac_per_burst{array}-length(scaf_units{array})/length(frac_per_unit{array}));
    
    % plot datapoints for array
    scatter(array*ones(1,length(frac_per_unit{array}(non_scaf_units{array}))), ...
        frac_per_unit{array}(non_scaf_units{array}), 200, "r", ".", 'jitter','on', 'jitterAmount',0.15);
    
    % plot datapoints for array
    sh1 = scatter(array*ones(1,length(frac_per_unit{array}(scaf_units{array}))), ...
        frac_per_unit{array}(scaf_units{array}), 200, "b", ".", 'jitter','on', 'jitterAmount',0.15);
    
end % array

% adjust axes
ylim([0,1])
xlim([0,length(frac_per_unit)+1])

% add axis labels
ylabel("Fraction of bursts")
xticks(1:length(frac_per_unit));
xticklabels(rec_names)

ax = gca;
ax.FontSize = 14;
ax.LineWidth = 3;
box off

% initiate subplot with fraction of scaffold units
subplot(2,1,2)
hold on

% plot number of scaffold units per array
hb = bar(bar_data);

% add error bars
ngroups = size(bar_data, 1);
nbars = size(bar_data, 2);

% Calculating the width for each bar group
groupwidth = min(0.8, nbars/(nbars + 1.5));
for array = 2:nbars
    x = (1:ngroups) - groupwidth/2 + (2*array-1) * groupwidth / (2*nbars);
    errorbar(x, bar_data(:,array), bar_error(:,array), 'k.');
end

% adjust colors
hb(1).FaceColor = [0 0 1];
hb(2).FaceColor = [1 0 0];
hb(1).FaceAlpha = .5;
hb(2).FaceAlpha = .5;

% add axis labels
ylabel("Frac. units per burst (+/- STD)")
xticks(1:length(frac_per_unit));
xticklabels(rec_names)

% adjust axes
ax = gca;
ax.FontSize = 14;
ax.LineWidth = 3;
box off


%% plot burst stats
% (Fig 1D-E, S2C-D)
% 1D-E = Load organoid Or1-4 recordings only
% S2C-D = Load all recordings
% Adjust figure size and y-axis range of second subplot to number of recordings included 

% initiate figure
fig = figure(4);
clf

% adjust size of figure
set(gcf,'PaperPositionMode','auto')
set(fig, 'Position', [100 100 1400 600])
set(fig, 'Renderer', 'painters')

% initate empty results arrays
burst_widths = []; 
burst_width_labels = [];
peak_times = cell(1,length(edges));%[];
num_bursts = NaN(1,length(edges));

% for each array
for array = 1:length(edges)

    % obtain burst widths
    burst_widths = [burst_widths, (edges{array}(:,2)-edges{array}(:,1))'];
    burst_width_labels = [burst_width_labels, array*ones(1,size(edges{array},1))];
    
    % select act times above thresh
    act_times_above = act_times{array}(above_thresh{array});
    peak_times{array} = act_times_above(~isnan(act_times_above));
        
    % count number of bursts
    num_bursts(array) = length(tburst{array});
    
end % array


% intiate subplot
subplot(2,1,1)

% plot burst width results per array
boxplot(burst_widths, burst_width_labels)

% add axis labels
ylabel("Burst width (ms)")

% adjust tick labels
xticks(1:length(rec_names))
xticklabels(rec_names)

ax = gca;
ax.YScale = "log";
ax.FontSize = 14;
ax.LineWidth = 3;
box off


% intiate subplot
subplot(2,1,2)
hold on 

% plot peak time results per array
violin(peak_times, 'facecolor', [1,1,1], 'mc', 'k', 'medc', []);
legend("off")

% add line for 0ms
yline(0, "r--", "LineWidth", 2);

% add axis labels
ylabel("Firing rate peak times (ms)")

% adjust tick labels
xticks(1:length(rec_names))
xticklabels(rec_names)

% adjust axis range
ylim([-2000,4000])
% ylim([-250,500])

ax = gca;
ax.FontSize = 14;
ax.LineWidth = 3;
box off


%% plot difference between 10 and 90 percentile of peak values
% (Fig 2G)
% 2G = Load organoid Or1-4 recordings only

% specify which percentile values to compute the difference between
PERC_DIFF = [10,90];

% initiate figure
fig = figure(5);
clf
hold on

% adjust size of figure
set(gcf,'PaperPositionMode','auto')
set(fig, 'Position', [100 100 400 300])
set(fig, 'Renderer', 'painters')

% for each array
for array = 1:length(act_times)
        
    % select sorted scaf units
    sorted_scaf = fliplr(mean_rate_ordering{array}(end-length(scaf_units{array})+1:end));
    
    % make empty result array
    peak_diff = NaN(1,length(sorted_scaf));
    
    % for each packet unit
    for unit = 1:length(sorted_scaf)
        
        % obtain all peak times
        peak_times_unit = act_times{array}(:,sorted_scaf(unit));
        
        % obtain percentile values
        prc_out = prctile(peak_times_unit, PERC_DIFF);
    
        % store difference
        peak_diff(unit) = diff(prc_out); % std(act_times{array}(:,sorted_scaf(unit))); % 
    
    end % unit
    
    % plot results
    plot(linspace(0,1,length(sorted_scaf)), peak_diff/max(peak_diff), "LineWidth", 2)

end % array

% add axis labels
xlabel("Sequence index (norm.)")
ylabel(sprintf("Diff. %.0f and %.0f perc. (norm.)", PERC_DIFF(1), PERC_DIFF(2)))

% add legend
legend(rec_names, "location", "North", "NumColumns", 4)
legend("boxoff")

box off
set(gca,'linewidth',3)
ax = gca;
ax.FontSize = 14;


%% compare average burst to burst correlation scaf vs non-scaf
% (Fig 2E, 6F, S1D)
% 2E = Load organoid L1, L2, L3 and L5 recordings only, set plot metric (line 595) to av_btb_corr_scores and choose ylabel (line 654, 655) accordingly
% 6F = Load mouse slice recordings (all) only, set plot metric (line 595) to av_btb_corr_scores and choose ylabel (line 654, 655) accordingly
% S1D = Load organoid recordings (all) only, set plot metric (line 595) to av_rate and choose ylabel (line 654, 655) accordingly, unset ylim (line 696), set yaxis to log (line 704)

plot_metric = av_btb_corr_scores; % av_rate

% only plot results for units with at least MIN_SPIKES in recording
MIN_SPIKES = 30;

% specify plot colors
box_colors = repmat(["r", "b"], 1,length(plot_metric));

% make empty result arrays
vectorized_data = [];
unit_type_group = [];
org_type_group = [];
boxplot_group = [];
tick_locs = zeros(1,length(plot_metric));


% for each array
for array = 1:length(plot_metric)
    
    % make copy of data
    plot_metric_values = plot_metric{array};
    
    % set scores for units with less than MIN_SPIKES to NaN
    plot_metric_values(spk_count{array} < MIN_SPIKES) = NaN;
    
    % store data
    vectorized_data = [vectorized_data, plot_metric_values(scaf_units{array}), plot_metric_values(non_scaf_units{array})];
    
    % store unit type labels
    unit_type_group = [unit_type_group, ones(1,length(scaf_units{array})), zeros(1,length(non_scaf_units{array}))];
    org_type_group = [org_type_group, array*ones(1,length(plot_metric_values))];
    boxplot_group = [boxplot_group, (1+(array-1)*3)*ones(1,length(scaf_units{array})), (2+(array-1)*3)*ones(1,length(non_scaf_units{array}))];
    
    % store tick location
    tick_locs(array) = 1.5+(array-1)*2;        
        
end % array

% intiate figure
fig = figure(6);
clf

% adjust size of figure
set(gcf,'PaperPositionMode','auto')
set(fig, 'Position', [100 100 500 300])
set(fig, 'Renderer', 'painters')

% plot boxplot of consistency scores
boxplot(vectorized_data, boxplot_group);

% obtain individual boxes
bh = findobj(gca,'Tag','Box');

% color individual boxes
for j=1:length(bh)
    patch(get(bh(j),'XData'),get(bh(j),'YData'),box_colors(j),'FaceAlpha',.5);
end

% add y-axis label
ylabel("Av. burst to burst corr.")
% ylabel("Av. firing rate (Hz)")
ylim([0,1])

% adjust ticks
xticks(tick_locs)
xticklabels(rec_names)

set(gca,'linewidth',3)
set(gca, 'fontsize', 14)
% set(gca, 'YScale', 'log')
box off


%% compare correlation scores
% (Fig 3E-F, 6G)
% 3E-F = Load organoid Or1-4 recordings only
% 6G = Load mouse slice recordings (all) only

% only plot results for units with at least MIN_SPIKES in recording
MIN_SPIKES = 30;
BIN_EDGES = linspace(0,1,21); 

% define bin centers
bin_centers = BIN_EDGES(2:end)-0.5*diff(BIN_EDGES);

% define log-normal distribution function
modelfun = @(b,x) 1./(x(:,1) .* b(1) .* sqrt(2 .* pi)) .* exp(-1 .* ( ( log(x(:,1)) - b(2) ).^2 / ( 2 .* b(1).^2 ) )) ;
est_initial = [0.8, -4];
xFitted = linspace(min(bin_centers), max(bin_centers), 301)';

% specify plot colors
box_colors = repmat(["r", "g", "b", "w"], 1,length(all_pw_corr_vals));

% make empty result arrays
unit_type_group = [];
org_type_group = [];
boxplot_group = [];
sep_labels = [];
tick_locs = zeros(1,length(all_pw_corr_vals));
violin_results = cell(1,4*length(all_pw_corr_vals));

% for each array
for array = 1:length(all_pw_corr_vals)
    
    % make copy of data
    x_corr_vals = all_pw_corr_vals{array};
    
    % set diagonal values to NaN
    x_corr_vals(logical(eye(size(x_corr_vals,1)))) = NaN;
    
    % set values for units with less than MIN_SPIKES to NaN
    x_corr_vals(spk_count{array} < MIN_SPIKES, :) = NaN;
    x_corr_vals(:, spk_count{array} < MIN_SPIKES) = NaN;
    
    % obtain selection of xcorr values that are correlated
    nz_units = find(sum(x_corr_vals, "omitnan")>0);
    scaf_nz_units = intersect(nz_units,scaf_units{array})';
    non_scaf_nz_units = nz_units;
    non_scaf_nz_units(ismember(non_scaf_nz_units,scaf_nz_units)) = [];
    
    % obtain correlation values per group
    scaf_vs_scaf = x_corr_vals(scaf_nz_units, scaf_nz_units);
    scaf_vs_scaf = scaf_vs_scaf(:);
    scaf_vs_scaf(isnan(scaf_vs_scaf)) = [];
    
    scaf_vs_non_scaf1 = x_corr_vals(scaf_nz_units, non_scaf_nz_units);
    scaf_vs_non_scaf2 = x_corr_vals(non_scaf_nz_units, scaf_nz_units);
    scaf_vs_non_scaf = [scaf_vs_non_scaf1(:); scaf_vs_non_scaf2(:)];
    scaf_vs_non_scaf(isnan(scaf_vs_non_scaf)) = [];
    
    non_scaf_vs_non_scaf = x_corr_vals(non_scaf_nz_units, non_scaf_nz_units);
    non_scaf_vs_non_scaf = non_scaf_vs_non_scaf(:);
    non_scaf_vs_non_scaf(isnan(non_scaf_vs_non_scaf)) = [];
    
    
    % store data
    violin_results{1+(array-1)*4} = scaf_vs_scaf;
    violin_results{2+(array-1)*4} = scaf_vs_non_scaf;
    violin_results{3+(array-1)*4} = non_scaf_vs_non_scaf;
    violin_results{4+(array-1)*4} = -100;
    
    % store separate label for white space
    sep_labels = [sep_labels, (4+(array-1)*4)];
    
    % store tick location
    tick_locs(array) = 2+(array-1)*4;
    
end % array

% intiate figure
fig = figure(7);
clf

% adjust size of figure
set(gcf,'PaperPositionMode','auto')
set(fig, 'Position', [100 100 800 300])
set(fig, 'Renderer', 'painters')

% initiate subplot
subplot(1,3,[1,2])

% specify violin plot colors
violin_color_mat = repmat([0,0,1 ; 0,1,0 ; 1,0,0 ; 1,1,1], length(all_pw_corr_vals), 1);

% plot boxplot of consistency scores
violin(violin_results, 'mc','r','medc','','plotlegend',0,'facecolor',violin_color_mat,'bw',0.05);

% adjust y-axis 
ylim([0,1])
ylabel("Pairwise correlation")

% adjust x-ticks
xticks(tick_locs)
xticklabels(rec_names)

ax1 = gca;
ax1.FontSize = 14;
ax1.LineWidth = 3;
box off

% initiate subplot
subplot(1,3,3)

hold on

% for each array
for array = 1:length(x_corr_counts)

    % plot histcounts
    plot(x_corr_counts{array}, bin_centers, "LineWidth", 2)
        
end % array

% add legend
legend(rec_names, "Location", "NorthEast")
legend("boxoff")

% adjust axes
ax2 = gca;
ax2.FontSize = 14;
ax2.LineWidth = 3;
ax2.YAxis.Visible = 'off';
box off

% link axes both plots
linkaxes([ax1, ax2], 'y')


%% Plot cosine similarity per organoid
% (Fig 4E, S12D)
% 4E = Load organoid Or1-4 recordings only
% S12D = Load organoid recordings (all) only

% specify plot parameters
WIND_BEFORE = 0;
WIND_AFTER = 0;

% specify time to peak
TIME_TO_PEAK = -450;
NORM_WIND = scaf_windows; 

% initiate figure
cm = figure(8);
clf
hold on

% adjust size of figure
set(gcf,'PaperPositionMode','auto')
set(cm, 'Position', [0 20 500 300])
set(cm, 'Renderer', 'painters')

leg_selection = zeros(1,length(mean_cos_sim));

% for each organoid
for array = 1:length(mean_cos_sim)
    
    % select start and end time of plot window
    plot_start = ceil(NORM_WIND{array}(1))-ceil(diff(NORM_WIND{array})*WIND_BEFORE);
    plot_end = ceil(NORM_WIND{array}(2))+ceil(diff(NORM_WIND{array})*WIND_AFTER);
    
    % select start and end index for selecting data
    plot_start_i = plot_start-TIME_TO_PEAK;
    plot_end_i = plot_end-TIME_TO_PEAK;
    
    % set range between 0 and 1
    range_0_1 = linspace(0,1,length(mean_cos_sim{array}(plot_start_i:plot_end_i)));
    
    % select plot data
    plot_data = mean_cos_sim{array}(plot_start_i:plot_end_i);
    
    % rescale plot data
    plot_data_rescale = rescale(plot_data, 0, 1);
    
    % plot data 
    plot_h = plot(range_0_1, plot_data_rescale, "LineWidth", 2);
    
    % store line for legend
    leg_selection(array) = plot_h(1);
    
end % array

% determine start and end of window
total_winds = WIND_BEFORE+WIND_AFTER+1;
wind_start = WIND_BEFORE/total_winds;
wind_end = (WIND_BEFORE+1)/total_winds;

% add legend
legend(leg_selection, rec_names, "Location", "SouthEast", "NumColumns", 4)
legend("boxoff")

% adjust axes
xticks([])

ylabel("Av. burst similarity (norm.)")
xlabel("Backbone period")

set(gca, "FontSize", 14)
set(gca, "LineWidth", 3)


%% Plot PCA manifolds colored by burst peak
% (Fig 4G, S13)
% you will have to manually flip some axes to make the manifolds look most
% similar between the different columns
% 4F = Load organoid Or1 recording only
% S13 = Load organoid Or1-4 recordings only

% initiate result cell array
peak_rels = cell(1,length(sbsc));

% for each array
for array = 1:length(sbsc)

    % define burst range
    burst_range = burst_windows{array}(1):burst_windows{array}(end);
    
    % initiate peak_rel array
    peak_rel = ones(size(rate_mat{array},1), 1)*burst_windows{array}(end);
    
    % for each burst
    for burst = 1:length(tburst{array})
        
        % Set the values around each peak to time relative to burst peak
        peak_rel(tburst{array}(burst)+burst_windows{array}(1):...
            tburst{array}(burst)+burst_windows{array}(end)) = burst_range;
        
    end % burst
    
    % store results for array
    peak_rels{array} = peak_rel;
    
end % array

% initiate figure
cm = figure(9); 
clf
set(gcf,'PaperPositionMode','auto')
set(cm, 'Position', [0 20 1450 300*length(sbsc)])
set(cm, 'Renderer', 'painters')

% set colormap
colormap('hsv');

% for each array
for array = 1:length(sbsc)
    
    % compute axis limits based on the subspaces
    lim_x = [ min([min(sbsc{array}.all(:, 1)) min(sbsc{array}.scaff(:, 1)) min(sbsc{array}.nscaff(:, 1))])
        max([max(sbsc{array}.all(:, 1)) max(sbsc{array}.scaff(:, 1)) max(sbsc{array}.nscaff(:, 1))])];
    lim_y = [ min([min(sbsc{array}.all(:, 2)) min(sbsc{array}.scaff(:, 2)) min(sbsc{array}.nscaff(:, 2))])
        max([max(sbsc{array}.all(:, 2)) max(sbsc{array}.scaff(:, 2)) max(sbsc{array}.nscaff(:, 2))])];
    lim_z = [ min([min(sbsc{array}.all(:, 3)) min(sbsc{array}.scaff(:, 3)) min(sbsc{array}.nscaff(:, 3))])
        max([max(sbsc{array}.all(:, 3)) max(sbsc{array}.scaff(:, 3)) max(sbsc{array}.nscaff(:, 3))])];
    
    
    % plot manifold for all units
    c = subplot(length(sbsc),4,1+(array-1)*4);
    scatter(sbsc{array}.all(:, 1), sbsc{array}.all(:, 2), 5, peak_rels{array}, 'filled');
    c.XAxis.Exponent=0;
    c.YAxis.Exponent=0;
    clb = colorbar();
    caxis([burst_windows{array}(1) burst_windows{array}(end)]);
    xlabel(clb, "Time rel. to peak (ms)");
    xlabel(sprintf("PC1 (%.1f %s)", var_vals{array}.all(1), "%"));
    ylabel(sprintf("PC2 (%.1f %s)", var_vals{array}.all(2), "%"));
    xlim(lim_x);
    ylim(lim_y);
    zlim(lim_z);
    c.XAxis.Exponent=0;
    c.YAxis.Exponent=0;
    set(c, "FontSize", 14)
    set(c, "LineWidth", 3)
    box off
    
    % plot manifold for scaffolding units only
    c = subplot(length(sbsc),4,2+(array-1)*4);
    scatter(sbsc{array}.scaff(:, 1), sbsc{array}.scaff(:, 2), 5, peak_rels{array}, 'filled');
    c.XAxis.Exponent=0;
    c.YAxis.Exponent=0;
    clb = colorbar();
    caxis([burst_windows{array}(1) burst_windows{array}(end)]);
    xlabel(clb, "Time Relative to Burst Peak (ms)");
    xlabel(sprintf("PC1 (%.1f %s)", var_vals{array}.scaff(1), "%"));
    ylabel(sprintf("PC2 (%.1f %s)", var_vals{array}.scaff(2), "%"));
    xlim(lim_x);
    ylim(lim_y);
    zlim(lim_z);
    c.XAxis.Exponent=0;
    c.YAxis.Exponent=0;
    set(c, "FontSize", 14)
    set(c, "LineWidth", 3)
    box off
    
    % plot manifold for non scaffolding units only
    c = subplot(length(sbsc),4,3+(array-1)*4);
    scatter(sbsc{array}.nscaff(:, 1), sbsc{array}.nscaff(:, 2), 5, peak_rels{array}, 'filled');
    clb = colorbar();
    caxis([burst_windows{array}(1) burst_windows{array}(end)]);
    xlabel(clb, "Time Rel. to Burst Peak (ms)", "FontSize", 14);
    xlabel(sprintf("PC1 (%.1f %s)", var_vals{array}.nscaff(1), "%"));
    ylabel(sprintf("PC2 (%.1f %s)", var_vals{array}.nscaff(2), "%"));
    xlim(lim_x);
    ylim(lim_y);
    zlim(lim_z);
    c.XAxis.Exponent=0;
    c.YAxis.Exponent=0;
    set(c, "FontSize", 14)
    set(c, "LineWidth", 3)
    box off
    
    % plot cumsum of explained variance
    c = subplot(length(sbsc),4,4+(array-1)*4);
    hold on
    plot(linspace(0, 1, length(var_vals{array}.all)), cumsum(var_vals{array}.all), 'Color', 'k', "LineWidth", 2);
    plot(linspace(0, 1, length(var_vals{array}.scaff)), cumsum(var_vals{array}.scaff), 'Color', 'b', "LineWidth", 2);
    plot(linspace(0, 1, length(var_vals{array}.nscaff)), cumsum(var_vals{array}.nscaff), 'Color', 'r', "LineWidth", 2);    
    xlabel("Num. PCs (norm.)");
    ylabel("Cumsum exp. var. (%)");
    xlim([0,1]);
    ylim([0,100]);
    set(c, "FontSize", 14)
    set(c, "LineWidth", 3)
    box off
    
end % array


%% Plot cumulative sum of variance explained per PC
% (Fig 4G)
% 4G = Load organoid Or1-4 recordings only

% initiate figure
cm = figure(10);
clf
hold on

% adjust size of figure
set(gcf,'PaperPositionMode','auto')
set(cm, 'Position', [0 20 350 300])
set(cm, 'Renderer', 'painters')

% define marker types
markers   = {'o', '*', '+', 'x'};

% make empty result arrays
cumsums_all = cell(1,length(var_vals));
cumsums_scaff = cell(1,length(var_vals));
cumsums_nscaff = cell(1,length(var_vals));
axis_handles   = [];

% for each array
for array = 1:length(var_vals)
    
    % compute cumulative sums
    cumsums_all{array}    = cumsum(var_vals{array}.all);
    cumsums_scaff{array}  = cumsum(var_vals{array}.scaff);
    cumsums_nscaff{array} = cumsum(var_vals{array}.nscaff);
    
    % interpolate subgroup arrays due to them being different sizes
    cumsums_scaff{array}  = interp_diff(cumsums_all{array}, cumsums_scaff{array});
    cumsums_nscaff{array} = interp_diff(cumsums_all{array}, cumsums_nscaff{array});
    
    % plot results
    x_axis = linspace(0, 1, length(cumsums_scaff{array}));
    a = plot(x_axis, cumsums_scaff{array}, markers{array}, 'Color', 'blue');
    plot(x_axis, cumsums_nscaff{array}, markers{array}, 'Color', 'red');
    axis_handles = [axis_handles, a];
    
end

% add yline to mark 0
yline(0,"k--","LineWidth",2);

% add legend
l = legend(gca, axis_handles, rec_names, 'FontSize',14);
legend("boxoff")
l.Location='northeast';

% adjust axes
xlabel('PC (norm)');
ylabel('Cumsum. var. rel. to all');

set(gca, 'FontSize', 14);
set(gca, 'LineWidth', 3);
box off
    

% function to interpolate
function ysc = interp_diff(x, y)

    xdataResamp = linspace(y(1), y(end), numel(x)); 
    ydataResamp = linspace(y(1), y(end), numel(y)); 

    ysc = interp1(ydataResamp, y, xdataResamp);

    ysc = ysc - x.';
    
end    