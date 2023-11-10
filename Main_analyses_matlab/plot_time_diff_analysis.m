%% load data

% define path to data
% LOAD_PATH = [paths to folder with files]; % !!! adjust path and file names for loading data

% % specify which organoid to analyse
organoids = ["L2_8M", "L2_8M_4H" ; "L3_8M", "L3_8M_4H"];

% make empty cell arrays for results
tburst = cell(size(organoids));
above_thresh = cell(size(organoids));
frac_per_unit = cell(size(organoids));
scaf_units = cell(size(organoids));
non_scaf_units = cell(size(organoids));
rate_mat = cell(size(organoids));
mean_rate_ordering = cell(size(organoids));


% for each array
for organoid = 1:size(organoids,1)
    
    % for each time
    for time = 1:size(organoids,2)
                
        % load single recording data for organoid
        s_rec_met = load(strcat(LOAD_PATH, sprintf('%s_single_recording_metrics.mat', organoids(organoid, time))));
        
        % store results in cell array
        tburst{organoid, time} = s_rec_met.tburst;
        above_thresh{organoid, time} = s_rec_met.above_thresh;
        frac_per_unit{organoid, time} = s_rec_met.frac_per_unit;
        scaf_units{organoid, time} = s_rec_met.scaf_units;
        non_scaf_units{organoid, time} = s_rec_met.non_scaf_units;
        rate_mat{organoid, time} = s_rec_met.rate_mat;
        mean_rate_ordering{organoid, time} = s_rec_met.mean_rate_ordering; 
        
    end % time
end % array

% % set organoid names
organoid_names = ["L2", "L3"];


%% compute fraction of overlapping scaffold units for confusion matrix
% (Fig S6A table values)

% initiate empty result arrays
both_scaf = zeros(1,size(organoids,1));
scaf_only_0h = zeros(1,size(organoids,1));
scaf_only_4h = zeros(1,size(organoids,1));
both_non_scaf = zeros(1,size(organoids,1));
    
% for each array
for array = 1:size(organoids,1)
    
    % compute fraction of overlapping units
    both_scaf(array) = sum(frac_per_unit{array,1}==1 & frac_per_unit{array,2}==1)/length(frac_per_unit{array,1});
    scaf_only_0h(array) = sum(frac_per_unit{array,1}==1 & frac_per_unit{array,2}<1)/length(frac_per_unit{array,1});
    scaf_only_4h(array) = sum(frac_per_unit{array,1}<1 & frac_per_unit{array,2}==1)/length(frac_per_unit{array,1});
    both_non_scaf(array) = sum(frac_per_unit{array,1}<1 & frac_per_unit{array,2}<1)/length(frac_per_unit{array,1});
    
end % array


%% compute correlation per burst over all bursts

BURST_RANGE = [250,500];
MIN_BURST_FRAC = 0.3;
MAXLAG = 30;

% make empty result cell arrays
xcorr_mat_00 = cell(1,size(organoids,1));
xcorr_mat_04 = cell(1,size(organoids,1));
xcorr_mat_40 = cell(1,size(organoids,1));
xcorr_mat_44 = cell(1,size(organoids,1));
av_per_unit_00 = cell(1,size(organoids,1));
av_per_unit_04 = cell(1,size(organoids,1));
av_per_unit_44 = cell(1,size(organoids,1));

% for each array
for array = 1:size(organoids,1)
    
    % set values with frac_per_unit smaller than MIN_BURST_FRAC to NaN
    unit_mask = frac_per_unit{array,1} < MIN_BURST_FRAC | frac_per_unit{array,2} < MIN_BURST_FRAC; 
    
    % compute matrix between 0h bursts
    xcorr_mat_00{array} = compute_burst_xcorr(rate_mat{array,1}, rate_mat{array,1}, tburst{array,1}, tburst{array,1}, ...
        above_thresh{array,1}, above_thresh{array,1}, BURST_RANGE, MAXLAG);
    av_per_unit_00{array} = mean(xcorr_mat_00{array}, [2,3], "omitnan");
    av_per_unit_00{array}(unit_mask) = NaN;
    
    % compute matrix between 0h bursts and 4h bursts
    xcorr_mat_04{array} = compute_burst_xcorr(rate_mat{array,1}, rate_mat{array,2}, tburst{array,1}, tburst{array,2}, ...
        above_thresh{array,1}, above_thresh{array,2}, BURST_RANGE, MAXLAG);
    av_per_unit_04{array} = mean(xcorr_mat_04{array}, [2,3], "omitnan");
    av_per_unit_04{array}(unit_mask) = NaN;

    xcorr_mat_40{array} = permute(xcorr_mat_04{array},[1,3,2]);
    
    % compute matrix between 4h bursts
    xcorr_mat_44{array} = compute_burst_xcorr(rate_mat{array,2}, rate_mat{array,2}, tburst{array,2}, tburst{array,2}, ...
        above_thresh{array,2}, above_thresh{array,2}, BURST_RANGE, MAXLAG);
    av_per_unit_44{array} = mean(xcorr_mat_44{array}, [2,3], "omitnan");
    av_per_unit_44{array}(unit_mask) = NaN;

end % array


%% plot burst to burst similarity
% (Fig S6B-E requires running previous cell)
% Fig S6B = Set UNIT_OI to 2
% Fig S6C = Set UNIT_OI to 62

ARRAY_OI = 1; % array index from which to select example unit
UNIT_OI = 2; % index of example unit to use

plot_unit = mean_rate_ordering{ARRAY_OI,1}(end-(UNIT_OI-1));

% initiate figure
fig = figure(1);
clf

% adjust size of figure
set(fig,'PaperPositionMode','auto')
set(fig, 'Position', [100 100 950 400])
set(fig, 'Renderer', 'painters')


% initiate subplot for single unit example
subplot(1,2,1)

% merge corr scores for different burst type comparisons
sim_matrix = [squeeze(xcorr_mat_00{ARRAY_OI}(plot_unit,:,:)), squeeze(xcorr_mat_04{ARRAY_OI}(plot_unit,:,:)) ;...
    squeeze(xcorr_mat_40{ARRAY_OI}(plot_unit,:,:)), squeeze(xcorr_mat_44{ARRAY_OI}(plot_unit,:,:))];

% plot results
imagesc(sim_matrix)

% add colorbar
cb = colorbar;
caxis([0,1])
ylabel(cb, "Norm. Cross. Corr.", "FontSize", 14)

% add indicators different recordigns
xline(size(xcorr_mat_00{ARRAY_OI},2)+0.5, "r--", "LineWidth", 4);
yline(size(xcorr_mat_00{ARRAY_OI},2)+0.5, "r--", "LineWidth", 4);

% adjust tick labels
xticks([1,size(xcorr_mat_00{ARRAY_OI},2),size(xcorr_mat_00{ARRAY_OI},2)+size(xcorr_mat_44{ARRAY_OI},2)])
xticklabels([1,size(xcorr_mat_00{ARRAY_OI},2),size(xcorr_mat_44{ARRAY_OI},2)])
yticks([1,size(xcorr_mat_00{ARRAY_OI},2),size(xcorr_mat_00{ARRAY_OI},2)+size(xcorr_mat_44{ARRAY_OI},2)])
yticklabels([1,size(xcorr_mat_00{ARRAY_OI},2),size(xcorr_mat_44{ARRAY_OI},2)])

% add axis labels
xlabel("Burst")
ylabel("Burst")

ax = gca;
ax.FontSize = 14;



% initiate subplot for all consistent scaf units example
subplot(2,2,2)
hold on

% initiate empty result array
boxplot_data = [];
boxplot_groups = [];
tick_locs = NaN(1,size(organoids,1));

% for each array
for array = 1:size(organoids,1)
    
    % obtain all index values for different scaffold types
    all_scaf_either = unique([scaf_units{array,1}; scaf_units{array,2}]);
    all_scaf_both = intersect(scaf_units{array,1}, scaf_units{array,2});
    all_scaf_single = all_scaf_either(~ismember(all_scaf_either,all_scaf_both));
    all_scaff_neither = intersect(non_scaf_units{array,1}, non_scaf_units{array,2});
    
    % plot data points
    scatter((1+3*(array-1))*ones(1,length(all_scaf_both)), av_per_unit_00{array}(all_scaf_both), 24, "k", "o", "filled", 'jitter', 'on', 'jitterAmount',0.1)
    scatter((2+3*(array-1))*ones(1,length(all_scaf_both)), av_per_unit_04{array}(all_scaf_both), 24, "k", "o", "filled", 'jitter', 'on', 'jitterAmount',0.1)
    scatter((3+3*(array-1))*ones(1,length(all_scaf_both)), av_per_unit_44{array}(all_scaf_both), 24, "k", "o", "filled", 'jitter', 'on', 'jitterAmount',0.1)
    
    % select boxplot data for correlations of scaffold units in both
    boxplot_data = [boxplot_data, av_per_unit_00{array}(all_scaf_both)', av_per_unit_04{array}(all_scaf_both)', ...
        av_per_unit_44{array}(all_scaf_both)'];
    
    % store boxplot group labels
    boxplot_groups = [boxplot_groups, (1+3*(array-1))*ones(1,length(all_scaf_both)), ...
        (2+3*(array-1))*ones(1,length(all_scaf_both)), ...
        (3+3*(array-1))*ones(1,length(all_scaf_both))];

    % store tick location
    tick_locs(array) = 2+3*(array-1);
    
end % array

% plot results
boxplot(boxplot_data, boxplot_groups, 'symbol', '')

% color boxplots
colors = [135/235,206/235,1; 65/225,105/225,1; 0,0,1; 135/235,206/235,1; 65/225,105/225,1; 0,0,1];

h = findobj(gca,'Tag','Box');
for j=1:length(h)
    patch(get(h(j),'XData'),get(h(j),'YData'),colors(j,:),'FaceAlpha',.5);
end

% adjust axes
ylim([0.9*min(boxplot_data), 1])
ylabel("Av corr. per unit")
xticks(tick_locs)
xticklabels(organoid_names)

ax = gca;
ax.FontSize = 14;


% initiate subplot for all consistent scaf units example
subplot(2,2,4)
hold on

% initiate empty result array
boxplot_data = [];
boxplot_groups = [];
tick_locs = NaN(1,size(organoids,1));

% for each array
for array = 1:size(organoids,1)
    
    % obtain all index values for different scaffold types
    all_scaf_either = unique([scaf_units{array,1}; scaf_units{array,2}]);
    all_scaf_both = intersect(scaf_units{array,1}, scaf_units{array,2});
    all_scaf_single = all_scaf_either(~ismember(all_scaf_either,all_scaf_both));
    all_scaff_neither = intersect(non_scaf_units{array,1}, non_scaf_units{array,2});
    
    % plot data points
    scatter((1+3*(array-1))*ones(1,length(all_scaf_both)), av_per_unit_04{array}(all_scaf_both), 24, "k", "o", "filled", 'jitter', 'on', 'jitterAmount',0.1)
    scatter((2+3*(array-1))*ones(1,length(all_scaf_single)), av_per_unit_04{array}(all_scaf_single), 24, "k", "o", "filled", 'jitter', 'on', 'jitterAmount',0.1)
    scatter((3+3*(array-1))*ones(1,length(all_scaff_neither)), av_per_unit_04{array}(all_scaff_neither), 24, "k", "o", "filled", 'jitter', 'on', 'jitterAmount',0.1)
    
    % select boxplot data for correlations of scaffold units in both
    boxplot_data = [boxplot_data, av_per_unit_04{array}(all_scaf_both)', av_per_unit_04{array}(all_scaf_single)', ...
        av_per_unit_04{array}(all_scaff_neither)'];
    
    % store boxplot group labels
    boxplot_groups = [boxplot_groups, (1+3*(array-1))*ones(1,length(all_scaf_both)), ...
        (2+3*(array-1))*ones(1,length(all_scaf_single)), ...
        (3+3*(array-1))*ones(1,length(all_scaff_neither))];

    % store tick location
    tick_locs(array) = 2+3*(array-1);
    
end % array

% plot results
boxplot(boxplot_data, boxplot_groups, 'symbol', '')

% color boxplots
colors = [1,0,0;0,1,0;0,0,1;1,0,0;0,1,0;0,0,1];

h = findobj(gca,'Tag','Box');
for j=1:length(h)
    patch(get(h(j),'XData'),get(h(j),'YData'),colors(j,:),'FaceAlpha',.5);
end

% adjust axes
ylim([0.9*min(boxplot_data), 1])
ylabel("Av corr. per unit")
xticks(tick_locs)
xticklabels(organoid_names)

ax = gca;
ax.FontSize = 14;


%% functions

function result_x_corr_vals = compute_burst_xcorr(data_r, data_c, peak_r, peak_c, ...
    above_thresh_r, above_thresh_c, burst_range, maxlag)

    % initiate result array
    result_x_corr_vals = NaN(size(data_r,2), length(peak_r), length(peak_c));

    % for each unit
    for unit = 1:size(data_r,2)

        % for each row burst
        for burst_r = 1:length(peak_r)

            % if unit has at least two spikes in burst_r
            if above_thresh_r(unit,burst_r)
                
                % select burst peak time
                peak_time_r = peak_r(burst_r);

                % select firing rate
                rate_r = data_r(peak_time_r-burst_range(1):peak_time_r+burst_range(2),unit);

                % for each column burst
                for burst_c = 1:length(peak_c)

                    % if unit has at least two spikes in burst_c
                    if above_thresh_c(unit,burst_c)
                
                        % select burst peak time
                        peak_time_c = peak_c(burst_c);

                        % select firing rate
                        rate_c = data_c(peak_time_c-burst_range(1):peak_time_c+burst_range(2),unit);

                         % compute cross correlation
                        [corr_r, ~] = xcorr(rate_c, rate_r, maxlag, 'coeff');

                        % obtain maximum correlation
                        [max_corr, ~] = max(corr_r);

                        % store results in matrices
                        result_x_corr_vals(unit, burst_r, burst_c) = max_corr;

                    end % if
                end % burst_c
            end % if
        end % burst_r
    end % unit
    
end % fun compute_burst_xcorr