%% load data

load('single_recording_metrics.mat')
load('single_recording_metrics_rand.mat')
 

%% plot raster plot with multi unit activity 
% (Fig 1A, 6A, S3A, S7A, S9A, S22A)
% 1A = Or1 with PLOT_RANGE set to [48000, 63500]
% 6A = M2S2 with PLOT_RANGE set to [190000, 240000]
% S3A = Or5 with PLOT_RANGE set to [870000, 930000]
% S7A = Or1 (randomized data, see line 17-18) with PLOT_RANGE set to [48000, 63500]
% S9A = Pr1 with PLOT_RANGE set to [23000, 31000]
% S22A = M3S1 with PLOT_RANGE set to [87000, 120000]

% select random spike times for plotting (uncomment for random data, overwrites non-random data)
% spk_times = spk_times_rand;
% spk_times_id = spk_times_id_rand;

% define parameters
PLOT_RANGE = [48000, 63500]; % select range for example plot in ms 
BURSTS_OI = 1:size(edges,1); % highlight bursts in figure

% initiate figure
fig = figure(1);
clf
hold on

% adjust size of figure
set(gcf,'PaperPositionMode','auto')
set(fig, 'Position', [10 10 1050 400])
set(fig, 'Renderer', 'painters')

% set color for both y-axes
left_color = [0.3010 0.7450 0.9330]; % light blue
right_color = [1 0 0]; % red
set(fig,'defaultAxesColorOrder',[left_color; right_color]);

% intiate left y-axis
yyaxis left

% for each burst of interest
for burst = 1:length(BURSTS_OI)
    
    % mark area of burst on the array
    area_h = area([edges(BURSTS_OI(burst),1), edges(BURSTS_OI(burst),2)], ...
        [length(spk_times)+1, length(spk_times)+1], "FaceColor", [0.5,0.5,0.5], ...
        "EdgeColor", [0.5,0.5,0.5], "FaceAlpha", 0.25, "EdgeAlpha", 0.25);
        
end % burst

% plot rasterplot
plot(vertcat(spk_times{:}), horzcat(spk_times_id{:}), '.', 'Color', [0.3010 0.7450 0.9330])

% do not display axis outline
box off

% adjust axes
xlim(PLOT_RANGE)
xticks([])

ylim([1,length(spk_times)])
yticks([1,length(spk_times)])
yticklabels([length(spk_times), 1])
ylabel("Unit")

ax = gca;
ax.FontSize = 14;
set(gca,'linewidth',3)

% initiate right y-axis
yyaxis right
hold on

% plot population firing rate
plot(1:length(pop_rate), pop_rate, '-r', "LineWidth", 2)

% set axis labels
ylabel('Population rate (kHz)')

% define axis limits


set(fig,'defaultAxesColorOrder',[left_color; right_color]);
ax = gca;
ax.FontSize = 14;


%% plot firing rate plot with multi unit activity for selection of recording
% (Fig 1B, 2A, 6B, S3B, S7B, S9B, S22B) 
% 1B = Or1 with PLOT_RANGE set to [48000, 63500] and ORDERING set to NaN
% 2A = Or1 with PLOT_RANGE set to [48000, 63500]
% 6B = M2S2 with PLOT_RANGE set to [190000, 240000]
% S3B = Or5 with PLOT_RANGE set to [870000, 930000]
% S7B = Or1 (randomized data, see line 105) with PLOT_RANGE set to [48000, 63500]
% S9B = Pr1 with PLOT_RANGE set to [23000, 31000]
% S22B = M3S1 with PLOT_RANGE set to [87000, 120000]

% define parameters
PLOT_RANGE = [48000, 63500]; % select range for example plot in ms
ORDERING = mean_rate_ordering; % unit ordering method, set to NaN for no ordering
SHOW_COLORBAR = false; % whether colorbar should be plotted

% make copy of data to plot
rate_mat_copy = rate_mat; % select rate_mat_rand for random data

% order units according to ORDERING
if ~isnan(ORDERING)
    rate_mat_copy = rate_mat_copy(:,ORDERING);
end % if

% initiate figure
fig = figure(2);
clf
hold on

% adjust size of figure
set(gcf,'PaperPositionMode','auto')
set(fig, 'Position', [10 10 1050 400])
set(fig, 'Renderer', 'painters')
hold on

% plot firing rates
imagesc(rate_mat_copy')
colormap hot
caxis([0,80])

% plot colorbar
if SHOW_COLORBAR
    cb = colorbar;
    ylabel(cb, "Firing rate (Hz)", "FontSize", 14)
end % if

% mark backbone units
if ~isnan(ORDERING)
    yline(size(rate_mat_copy,2)-length(scaf_units)+0.5, "g--", "LineWidth", 4);
end % if

% define axis limits
xlim(PLOT_RANGE)
ylim([1,size(rate_mat_copy,2)])

% do not display axis outline
box off

% adjust axes
yticks([1,size(rate_mat_copy,2)])
yticklabels([size(rate_mat_copy,2), 1])
xticks([])

ax = gca;
ax.FontSize = 14;


%% plot firing rate plot with multi unit activity for single burst
% (Fig 1B, 2B)
% 1B = Or1 bursts 13, 14, 15 and 16 with ORDERING set to NaN
% 2B = Or1 bursts 13, 14, 15 and 16 

% define parameters
BURST_OI = 13; % which burst to plot
WINDOW = [250, 500]; % window around burst peak to plot
ORDERING = mean_rate_ordering; % unit ordering method, set to NaN for no ordering

% set PLOT_RANGE based on BURST_OI and WINDOW
PLOT_RANGE = [tburst(BURST_OI)-WINDOW(1), tburst(BURST_OI)+WINDOW(2)]; % select range for plot in ms

% if units should be ordered
if ~isnan(ORDERING)
    
    % order data
    rate_mat_ordered = rate_mat(:,ORDERING);
    
    % order units according to ORDERING
    cut_rate_mat = rate_mat_ordered(PLOT_RANGE(1):PLOT_RANGE(2),end-length(scaf_units)+1:end);
    
else
    
    % select units with at least 2 spikes in the burst
    cut_rate_mat = rate_mat(PLOT_RANGE(1):PLOT_RANGE(2),above_thresh(:,BURST_OI));
    
    % select peak act times for these units
    act_time_selection = act_times(BURST_OI,above_thresh(:,BURST_OI));
    
    % order act times selection
    [~,order_i] = sort(act_time_selection,"descend");
    
    % order units based on their average peak time
    cut_rate_mat = cut_rate_mat(:,order_i);
    
end % if

% initiate figure
fig = figure(3);
clf
hold on

% adjust size of figure
set(gcf,'PaperPositionMode','auto')
set(fig, 'Position', [10 10 250 400])
set(fig, 'Renderer', 'painters')

hold on

% plot firing rates
imagesc(cut_rate_mat')
colormap hot
caxis([0,80])

% do not display axis outline
box off

% set axis label
ylim([0.5,size(cut_rate_mat,2)+0.5])
yticks([1,size(cut_rate_mat,2)])
yticklabels([size(cut_rate_mat,2), 1])
xticks([WINDOW(1)+1])
xticklabels("P")

% adjust axes
ax = gca;
ax.FontSize = 14;


%%  Plot figure for activation sequences
% (Fig 2F)
% 2F = Or1

% specify which percentile values to compute the difference between
PERC_DIFF = [10,90];

% initiate figure
fig = figure(4);
clf

% adjust size of figure
set(gcf,'PaperPositionMode','auto')
set(fig, 'Position', [100 100 500 300])
set(fig, 'Renderer', 'painters')
hold on

% select peak act times for scaffold units and order them
act_time_selection = act_times;
act_time_selection(above_thresh == 0) = NaN;
act_time_selection = act_time_selection(:,fliplr(mean_rate_ordering));
act_time_selection = act_time_selection(:,1:length(scaf_units));
unit_frac_selection = frac_per_unit(fliplr(mean_rate_ordering));

% make empty result matrix and cell array
per_unit_cell = cell(1,size(act_time_selection,2));
per_unit_perc = zeros(size(act_time_selection,2), 2);
per_unit_med = zeros(size(act_time_selection,2), 1);

% for each scaffold unit
for unit = 1:size(act_time_selection,2)
    
    % store act times in cell array
    per_unit_cell{unit} = act_time_selection(:,unit);
    
    % compute percentile values for unit
    per_unit_perc(unit,:) = prctile(per_unit_cell{unit}, PERC_DIFF);
    
    % compute percentile values for unit
    per_unit_med(unit) = median(per_unit_cell{unit}, "omitnan");
    
end % unit

% plot scatter results
for unit = 1:length(per_unit_cell)
    scatter(unit*ones(1,length(per_unit_cell{unit})), per_unit_cell{unit}, 24, "k",...
        "o", "filled", 'jitter', 'on', 'jitterAmount', 0.2) 
end % unit

% plot median line
plot(1:length(per_unit_med), per_unit_med, "b", "LineWidth", 3)

% plot percentile lines
plot(1:size(per_unit_perc,1), per_unit_perc(:,1), "r", "LineWidth", 3)
plot(1:size(per_unit_perc,1), per_unit_perc(:,2), "r", "LineWidth", 3)

% adjust axes
xlabel("Unit ID (ordered)")
ylabel("Peak time w.r.t. burst peak (ms)")
xlim([1,length(per_unit_cell)])

ax = gca;
ax.FontSize = 14;
ax.LineWidth = 3;
box off


%% plot burst peak centered mean rate for scaff and non scaff
% (Fig 2C, 6C, S3C, S7C, S9C, S22C) 
% 2C = Or1
% 6C = M2S2 
% S3C = Or5
% S7C = Or1 (randomized data, see line 302) 
% S9C = Pr1 
% S22C = M3S1

% select random rates for plotting (uncomment for random data, overwrites non-random data)
% av_rate = av_rate_rand;

% make empty array for pre-processing
av_rate_norm = NaN(length(scaf_units), size(av_rate,2));

% for each scaffold unit
for unit = 1:length(scaf_units)
    
    % specify which unit to select
    select_unit = mean_rate_ordering(end-(unit-1));
   
    % normalize rate between 0 and 1 and store
    av_rate_norm(end-(unit-1),:) = rescale(av_rate(select_unit,:), 0, 1);
    
end % unit

% initiate figure
fig = figure(5);
clf

% adjust size of figure
set(gcf,'PaperPositionMode','auto')
set(fig, 'Position', [100 100 350 500])
set(fig, 'Renderer', 'painters')
hold on

% plot mean rates for ARRAY_OI
imagesc(av_rate_norm)
axis xy

% mark burst peak time
xline(251,"g--", "LineWidth", 5, "Alpha", 1)

% add colorbar
cb = colorbar;
colormap hot
ylabel(cb, "Mean rate (norm.)", "FontSize", 14)

% adjust axes
ylabel("Backbone unit ID")
xticks([1,251,751])
xticklabels([-250, 0, 500])
yticks([1,length(scaf_units)])
yticklabels([length(scaf_units),1])
ylim([0.5, length(scaf_units)+0.5])

ax = gca;
ax.FontSize = 14;
ax.LineWidth = 3;
box off


%% plot burst peak centered spikes and rate plus btb corr single unit
% (Fig 2D, 6D, S7D)
% 2D = Or1 SORTED_UNIT_OI 7, 27 and 66
% 6D = M2S2 SORTED_UNIT_OI 12 and 126
% S7D = Or1 (randomized data, see line 361-363) SORTED_UNIT_OI 7, 27 and 66

% select random rates for plotting (uncomment for random data, overwrites non-random data)
% cut_spk_mat = cut_spk_mat_rand;
% all_btb_corr_scores = all_btb_corr_scores_rand;
% av_rate = av_rate_rand;

% specify unit to plot (index after sorting)
SORTED_UNIT_OI = 7;
    
% select actual unit number
flip_order = fliplr(mean_rate_ordering);
unit_i = flip_order(SORTED_UNIT_OI);

% initiate figure
fig = figure(6);
clf

% adjust size of figure
set(gcf,'PaperPositionMode','auto')
set(fig, 'Position', [100 100 300 500])
set(fig, 'Renderer', 'painters')

% initiate subplot
subplot(2,1,1)

% set color for both y-axes
left_color = [0.3010 0.7450 0.9330]; % light blue
right_color = [1 0 0]; % red
set(fig,'defaultAxesColorOrder',[left_color; right_color]);

yyaxis left
hold on

% for each burst
for burst = 1:size(cut_spk_mat{unit_i}, 1)
    
    % obtain spike locations
    spk_locs = find(cut_spk_mat{unit_i}(burst,:))-250;
    
    % for each spike
    for spk = 1:length(spk_locs)
        
        % plot spike
        plot([spk_locs(spk),spk_locs(spk)],[burst-1,burst], 'Color', [0.3010 0.7450 0.9330], "Marker", "none")
        
    end % spk
end % burst

% mark burst peak
xline(0,"g--", "LineWidth", 3);

% add axis label
ylabel("Burst")

ylim([0,size(edges,1)])

yyaxis right

% plot burst peak centered summed rate
plot(linspace(-250, 500, size(av_rate,2)), av_rate(unit_i, :), "Color", [1, 0, 0, 0.5], "LineWidth", 2)

% add axis label
ylabel("Mean rate (Hz)")

% adjust axes
xlim([-250, 500])
xlabel("Rel. time (ms)")
set(gca,'linewidth',3)
box off
ax = gca;
ax.FontSize = 14;

% intiate subplot
subplot(2,1,2)

% plot burst to burst correlation for unit
imagesc(squeeze(all_btb_corr_scores(unit_i, :, : )));

% add color bar
cb = colorbar;
colormap(viridis);
caxis([0,1])
ylabel(cb, "Burst to burst corr.", "FontSize", 14)

% add axis labels
xlabel("Burst")
ylabel("Burst")

% adjust axes
ax = gca;
ax.FontSize = 14;


%% plot results for pairwise correlation scores of whole culture
% (Fig 3D+S10, 6E)
% 3G+S10 = Or1
% 6E = M2S2 (only subplot 1 is needed)

MIN_SPIKES = 30;

% initiate figure
fig = figure(7);
clf

% adjust size of figure
set(gcf,'PaperPositionMode','auto')
set(fig, 'Position', [0 20 700 300])
set(fig, 'Renderer', 'painters')
 
% prepare plot data
corr_plot_data = all_pw_corr_vals(fliplr(mean_rate_ordering), fliplr(mean_rate_ordering));
corr_plot_data(spk_count(fliplr(mean_rate_ordering)) < MIN_SPIKES,:) = [];
corr_plot_data(:,spk_count(fliplr(mean_rate_ordering)) < MIN_SPIKES) = [];

lag_plot_data = all_pw_corr_lags(fliplr(mean_rate_ordering), fliplr(mean_rate_ordering));
lag_plot_data(spk_count(fliplr(mean_rate_ordering)) < MIN_SPIKES,:) = [];
lag_plot_data(:,spk_count(fliplr(mean_rate_ordering)) < MIN_SPIKES) = [];

% make subplot for correlation values
xcorr_h = subplot(1,2,1);

% plot results
imagesc(corr_plot_data)
axis ij

hold on

% mark scaffold units
yline(length(scaf_units)+0.5, "r", "LineWidth", 3);
xline(length(scaf_units)+0.5, "r", "LineWidth", 3);

% add colorbar
cb = colorbar;
colormap(viridis)
ylabel(cb, "Cross-correlation", "FontSize", 14)
caxis([0, 1])

% adjust axes
xlabel("Unit")
ylabel("Unit")
ax = gca;
ax.FontSize = 14;

% make subplot for lag values
xlagg_h = subplot(1,2,2);

% plot results 
imagesc(lag_plot_data)
axis ij

hold on

% mark scaffold units
yline(length(scaf_units)+0.5, "k", "LineWidth", 3);
xline(length(scaf_units)+0.5, "k", "LineWidth", 3);

% add colorbar
cb = colorbar;
colormap(xlagg_h, redblue)
ylabel(cb, "Lag (ms)", "FontSize", 14)
caxis([-50, 50])

% adjust axes
xlabel("Unit")
ylabel("Unit")
ax = gca;
ax.FontSize = 14;


%% Plot centered pop rate and burst similarity
% (Fig 1D, 4A-E)
% 1C = Or1 set PLOT_BURST_SIM to true
% 4A-E = Or1 set PLOT_BURST_SIM to false

% specify plot parameters
POP_RATE_THRESH = 0.03; % threshold for plotting window as fraction of pop_rate peak
PLOT_BURST_SIM = true; % whether burst similarity results should be included
FRAMES_OI = [-50,100]; % frames to highlight
COMP_DATA = {cos_sim_scaf, cos_sim_non_scaf}; % data for significance comparison, you can choose from cos_sim_scaf cos_sim_non_scaf cos_sim_rand

% initiate figure
fig = figure(8);
clf

% adjust size of figure
set(gcf,'PaperPositionMode','auto')
set(fig, 'Position', [100 100 1400 600])
set(fig, 'Renderer', 'painters')

% specify axis colors
left_color = [0 0 0];
right_color = [1 0 0]; 
set(fig,'defaultAxesColorOrder',[left_color; right_color]);

% initiate subplot for burst peak centered population rate
subplot(2,3,1)
hold on
        
% set frame range
frame_range = [-450:size(centered_pop_rate,2)-451];

% for each burst
for burst = 1:size(centered_pop_rate,1)
    
    % plot population rate
    plot(frame_range, centered_pop_rate(burst,:), "Color", [0.5, 0.5, 0.5, 0.5], "LineWidth", 0.5)
    
end % burst

% compute mean and std of centered_pop_rate
mean_centered_pop_rate = mean(centered_pop_rate);
std_centered_pop_rate = std(centered_pop_rate);

% plot mean and std over all bursts
plot(frame_range, mean_centered_pop_rate, "k", "LineWidth", 2)
plot(frame_range, mean_centered_pop_rate-std_centered_pop_rate, "k--", "LineWidth", 1)
plot(frame_range, mean_centered_pop_rate+std_centered_pop_rate, "k--", "LineWidth", 1)

% define x-axis range
above_thresh_i = find(mean_centered_pop_rate > max(mean_centered_pop_rate)*POP_RATE_THRESH);   

% add axis labels
ylabel("Pop. rate (kHz) +- STD")
xlabel("Time relative to burst peak (ms)")

% adjust axes
ylim_curr = ylim;
ylim([0,ylim_curr(2)])
xlim([frame_range(min(above_thresh_i)), frame_range(max(above_thresh_i))])

ax = gca;
set(ax,'FontSize',14)
set(ax,'linewidth',3)

% if burst similarity data should be plotted
if PLOT_BURST_SIM

    % initiate right yaxis
    yyaxis right
    hold on

    % plot number average cosine similarity
    plot(frame_range, mean_cos_sim, "LineWidth", 2, "Color", "r")

    % mark scaffold period
    area_h = area([scaf_window(1), scaf_window(2)], [1 1], "FaceColor", [0.3010 0.7450 0.9330], ...
        "EdgeColor", [0.3010 0.7450 0.9330], "FaceAlpha", 0.25, "EdgeAlpha", 0.25);

    % add marker lines for example figures
    xline(FRAMES_OI(1), "Color", "b", "LineStyle", "--", "LineWidth", 3); 
    xline(FRAMES_OI(2), "Color", "b", "LineStyle", "--", "LineWidth", 3);

    % add axis labels
    ylabel("Av. burst similarity")

    % adjust axes
    xlim([frame_range(min(above_thresh_i)), frame_range(max(above_thresh_i))])
    ylim([0,0.6])
    ax = gca;
    ax.YAxis(2).Color = "r";
    set(ax,'FontSize',14)
    set(ax,'linewidth',3)

    
    % initiate subplot for cosine similarity heatmap
    subplot(2,3,2)

    % plot heatmap
    plot_cos_sim_heatmap(cos_sim, frame_range, FRAMES_OI(1)) 

    
    % initiate subplot for PCA distribution
    subplot(2,3,3)

    % plot heatmap
    plot_cos_sim_heatmap(cos_sim, frame_range, FRAMES_OI(2)) 

    
    % initiate subplot for cosine similarity between two bursts over time
    subplot(2,3,4)
    hold on

    % make empty result array
    all_pairwise = zeros((length(tburst)-2*(length(tburst)-3))/2, length(frame_range));

    % initiate counter
    counter = 1;

    % make list of all comparison bursts
    all_comp_bursts = 1:length(tburst);

    % for each burst
    for burst = 1:length(tburst)

        % remove burst from all_bursts list
        all_comp_bursts(all_comp_bursts == burst) = [];

        % for each comparison burst
        for comp_burst = 1:length(all_comp_bursts)

            % plot cosine similarity relative to burst peak
            plot(frame_range, squeeze(cos_sim(burst, all_comp_bursts(comp_burst), :)), "Color", [0, 0, 0, 0.05], "LineWidth", 0.5)

            % store pairwise cosine similarity
            all_pairwise(counter, :) = squeeze(cos_sim(burst, all_comp_bursts(comp_burst), :));

            % add 1 to counter
            counter = counter + 1;

        end % comp_burst

    end % burst
    
    % mark scaffold period
    area_h = area([scaf_window(1), scaf_window(2)], [1 1], "FaceColor", [0.3010 0.7450 0.9330], ...
        "EdgeColor", [0.3010 0.7450 0.9330], "FaceAlpha", 0.25, "EdgeAlpha", 0.25);

    ylim([0,1])
    xlim([frame_range(min(above_thresh_i)), frame_range(max(above_thresh_i))])
    
    % add axis labels
    ylabel("Pairwise burst similarity")
    xlabel("Time relative to burst peak (ms)")
    
    % initiate right yaxis
    yyaxis right
    hold on
    

    % plot variance of all pairwise cosine similarities
    plot(frame_range, std(all_pairwise), "Color", "r", "LineWidth", 2)
    
    % add axis label
    ylabel("STD burst similarity")

    set(gca,'FontSize',14)
    set(gca,'linewidth',3)
    
    
    % initiate subplot for burst similarity for scaf, non-scaf and rand
    subplot(2,3,5)
    hold on
    
    % plot average cosine similarity
    sc_h = plot(frame_range, mean_cos_sim_scaf, "k", "LineWidth", 2);
    nsc_h = plot(frame_range, mean_cos_sim_non_scaf, "k--", "LineWidth", 2);
    r_h = plot(frame_range, mean_cos_sim_rand, "k:", "LineWidth", 2);
    
    % make empty result array for significance scores
    sig_scores = zeros(1,length(mean_cos_sim_scaf));
    sig_val = zeros(1,length(mean_cos_sim_scaf));
    
    % for each frame
    for frame = 1:length(mean_cos_sim_scaf)
                
        % select data for frame
        frame_data_comp1 = COMP_DATA{1}(:,:,frame);
        frame_data_comp2 = COMP_DATA{2}(:,:,frame);
        
        % compute whether difference is significant
        [sig_scores(frame), pval] = ttest(frame_data_comp1(logical(tril(ones(size(frame_data_comp1)),-1))), ...
            frame_data_comp2(logical(tril(ones(size(frame_data_comp2)),-1))), "Alpha", 0.05/length(above_thresh_i));        
        
        sig_val(frame) = -log10(pval);
        
    end % frame
        
    % change inf to max value
    sig_val(sig_val==Inf) = 320;

    % add axis labels
    ylabel("Burst similarity")        
        
    % initiate right yaxis
    yyaxis right
    hold on
    
    % plot significane values
    plot(frame_range, sig_val, "r", "LineWidth", 2)
    
    % adjust yxes
    
    
    % add legend
    legend([sc_h, nsc_h, r_h], "Packet", "Non-packet", "Shuffled", "Location", "NorthOutside", "FontSize", 14, "NumColumns", 3)
    legend("boxoff")
    
    % adjust axes
    ylabel("-log10(P)")
    ylim([0,320])
    yticks([0,100,200,320])
    yticklabels(["0", "100", "200", ">320"]) 
    
    xlim([frame_range(min(above_thresh_i)), frame_range(max(above_thresh_i))])
    xlabel("Time relative to burst peak (ms)")
    
    ax = gca;
    ax.YAxis(2).Color = "r";
    ax.FontSize = 14;
    ax.LineWidth = 3;

end % if



%% compute average burst similarity relative to burst peak for subsets of neurons
% (Fig S12A, S12B)
% S12A = Or1 set SMALLER to false
% S12B = Or1 set SMALLER to true
% computations take some time so plotting is done is separate cell below this one

% compute average correlation
av_xcorr = mean(all_pw_corr_vals, "omitnan");

% normalize rate mat
z_rate_mat = zscore(rate_mat);

% set parameters
WINDOW = burst_window; % time relative to burst peak to consider
SMALLER = false;

if SMALLER == true
    PERC_RANGE = linspace(20,95,76);
else
    PERC_RANGE = linspace(5,80,76);
end

% make frame range relative to burst peak
frame_range = WINDOW(1):WINDOW(2);

% make empty result arrays
thresh_mean_cos_sim = zeros(length(PERC_RANGE),length(frame_range));
thresh_cos_sim = zeros(length(PERC_RANGE),length(tburst),length(tburst),length(frame_range));
centered_pop_rate = zeros(length(PERC_RANGE),length(tburst),length(frame_range));

% select what the selection should be based on
comp_oi = av_xcorr; 

% set 0 values to NaN
comp_oi(comp_oi == 0) = NaN;

% for each treshold
for thresh = 1:length(PERC_RANGE)
    
    thresh
    
    % compute percentile of interest
    perc_oi = prctile(comp_oi, PERC_RANGE(thresh));
    
    % make selection mask
    if SMALLER == true
        mask = comp_oi<perc_oi;
    else
        mask = comp_oi>perc_oi;
    end %if
    
    % select units for analysis
    temp_rate_mat = z_rate_mat(:,mask);
    

    % for each frame
    for frame = 1:length(frame_range)
        
        % make empty result matrix
        frame_rates = NaN(length(tburst), size(temp_rate_mat,2));
        
        % for each burst
        for burst = 1:length(tburst)  
            
            % obtain rate for frame and store
            frame_rates(burst, :) = temp_rate_mat(int32(tburst(burst)+frame_range(frame)), :);
            
            % for each comparison burst
            for comp_burst = 1:length(tburst) 
                
                % obtain rate for frame of comparison burst
                frame_rate_comp = temp_rate_mat(int32(tburst(comp_burst)+frame_range(frame)), :);

                % compute cosine similarity between the two vectors and store
                thresh_cos_sim(thresh,burst,comp_burst,frame) = dot(frame_rates(burst, :), ...
                    frame_rate_comp)/(norm(frame_rates(burst, :))*norm(frame_rate_comp));
                
            end % comp_burst
            
            % if this is the first frame
            if frame == 1
                
                % store population rate relative to burst
                centered_pop_rate(thresh,burst,:) = pop_rate(int32(tburst(burst)+frame_range(1)) : int32(tburst(burst)+frame_range(end)));
                
            end % if
            
        end % burst
        
        
        % compute average cosine similarity
        thresh_mean_cos_sim(thresh, frame) = mean(thresh_cos_sim(thresh,:,:,frame), "all", "omitnan");
        
    end % frame
    
end % thresh


%% plot results
% continuation from previous cell

% initiate figure
fig = figure(9);
clf

% adjust size of figure
set(gcf,'PaperPositionMode','auto')
set(fig, 'Position', [100 100 500 300])
set(fig, 'Renderer', 'painters')

% set colors
colors = jet(size(thresh_mean_cos_sim,1));

% set line styles for plotting
line_styles = ["-", "--", ":"];

if SMALLER == false
    colors = fliplr(colors);
end % if

hold on

for thresh = 1:size(thresh_mean_cos_sim,1) 
    
    % plot number average cosine similarity
    plot(frame_range, thresh_mean_cos_sim(thresh,:), "Color", colors(thresh,:), "LineWidth", 2)
    
end % thresh

% add colorbar
cb = colorbar;
colormap(jet)

caxis([PERC_RANGE(1),PERC_RANGE(end)])
if SMALLER == false
    caxis([100-PERC_RANGE(end),100-PERC_RANGE(1)]);
end % if

if SMALLER == true
    ylabel(cb, "% lowest values", "FontSize", 14)
else
    ylabel(cb, "% highest values", "FontSize", 14)
end

% add axis labels
ylabel("Average burst similarity")
xlabel("Time relative to burst peak (ms)")

ylim([0,1])

% adjust axes
xlim(WINDOW)
set(gca,'FontSize',14)
set(gca,'linewidth',3)
    

%% compute statistical difference with main
% (Fig S12C)
% S12C = Or1 
% First run cell for processing data of the previous figure (previous 2 
% cells). Then compute pvals with data in this cell and rename results to
% pval_smaller or pval_larger depending on whether SMALLER is true or false
% Do that for SMALLER set to both true and false (so that you have a
% pval_smaller and a pval_larger variable). Finally run the cell below this
% one to plot the figure.

THRESH_OI = 1; % which threshold from PERC_RANGE is compared to the last value

% make empty result array
pval = zeros(1,size(thresh_cos_sim, 4));

% for each frame
for frame = 1:size(thresh_cos_sim, 4)
    
    % compute statistics between thresh_oi and all
    [~,p] = ttest(reshape(thresh_cos_sim(1,:,:,frame),1,[]), reshape(thresh_cos_sim(end,:,:,frame),1,[]));
    
    % store -log10(p) value
    pval(frame) = -log10(p);
    
end % frame

% set out of range pvals to maximum value
pval(pval == Inf) = 320;


%% plot significance

% initiate figure
fig = figure(10);
clf

% adjust size of figure
set(gcf,'PaperPositionMode','auto')
set(fig, 'Position', [100 100 500 300])
set(fig, 'Renderer', 'painters')

hold on

% define frame range for plotting
frame_range = burst_window(1):burst_window(end);

% plot results
plot(frame_range, pval_smaller, "b", "LineWidth", 2)
plot(frame_range, pval_larger, "r", "LineWidth", 2)

% add legend
legend("Top 20% vs all", "Bottom 20% vs all", "Location", "South", "FontSize", 14, "NumColumns", 1)
legend("boxoff")

% adjust axes
xlim([frame_range(1), frame_range(end)])
xlabel("Time relative to burst peak (ms)")
ylabel("-log10(P)")
ylim([0,320])
yticks([0,100,200,320])
yticklabels(["0", "100", "200", ">320"])

ax = gca;
set(ax,'FontSize',14)
set(ax,'linewidth',3)
    

%% functions

function plot_cos_sim_heatmap(cos_sim, frame_range, frame_oi)

    % if a frame of interest is specified
    if ~isnan(frame_oi)
        
        % plot heatmap of cosine similarity during frame_oi
        imagesc(cos_sim(:,:,frame_range == frame_oi))

        % add title
        title(sprintf("%.0f ms relative to peak", frame_oi))
    
    else
        
        % plot heatmap of cosine similarity averaged over all frames
        imagesc(mean(cos_sim,3))

        % add title
        title("Av. burst similarity")
        
    end % if
    
    % add colorbar
    cb = colorbar;
    colormap(viridis)
    ylabel(cb, "Burst similarity", "FontSize", 14)
    caxis([0,0.6])

    % add axis labels
    xlabel("Burst")
    ylabel("Burst")

    set(gca,'FontSize',14)
    set(gca,'linewidth',3)
    
end % fun plot_cos_sim_heatmap