%% load data

% % specify which organoid to analyse
% rec_paths = [paths to folders with files]; % !!! adjust path and file names for loading data

% make empty cell arrays for results
av_rate = cell(1,length(rec_paths));
spk_count = cell(1,length(rec_paths));
scaf_units = cell(1,length(rec_paths));
non_scaf_units = cell(1,length(rec_paths));
av_btb_corr_scores = cell(1,length(rec_paths));
all_pw_corr_vals = cell(1,length(rec_paths));
var_vals = cell(1,length(rec_paths));

av_btb_corr_scores_rand = cell(1,length(rec_paths));
all_pw_corr_vals_rand = cell(1,length(rec_paths));
var_vals_rand = cell(1,length(rec_paths));

% for each array
for array = 1:length(rec_paths)
        
    % split file identifier
    split_name = split(rec_paths(array), "/");
    
    % % load data of first treatment
    
    % load single recording data for organoid
    s_rec_met = load(sprintf('%s/single_recording_metrics.mat', rec_paths(array)));
    s_rec_met_rand = load(sprintf('%s/single_recording_metrics_rand.mat', rec_paths(array)));
    
    % store results in cell array
    av_rate{array} = s_rec_met.spk_count / (size(s_rec_met.rate_mat,1)/1000);
    spk_count{array} = s_rec_met.spk_count;
    scaf_units{array} = s_rec_met.scaf_units;
    non_scaf_units{array} = s_rec_met.non_scaf_units;
    av_btb_corr_scores{array} = s_rec_met.av_btb_corr_scores;
    all_pw_corr_vals{array} = s_rec_met.all_pw_corr_vals;
    var_vals{array} = s_rec_met.vars;
    
    % store shuffled results in cell array
    av_btb_corr_scores_rand{array} = s_rec_met_rand.av_btb_corr_scores_rand;
    all_pw_corr_vals_rand{array} = s_rec_met_rand.all_pw_corr_vals_rand;
    var_vals_rand{array} = s_rec_met_rand.vars_rand;
    
end % array


%% set recording names
% rec_names = ["Or1", "Or2", "Or3", "Or4", "Or5", "Or6", "Or7", "Or8"];
% rec_names = ["M1S1", "M1S2", "M2S1", "M2S2", "M3S1", "M3S2"];
rec_names = ["Or1", "Or2", "Or3", "Or4", "Or5", "Or6", "Or7", "Or8", "M1S1", ...
    "M1S2", "M2S1", "M2S2", "M3S1", "M3S2", "Pr1", "Pr2", "Pr3", "Pr4", ...
    "Pr5", "Pr6", "Pr7", "Pr8", "Pr9", "Pr10"];


%% compare average burst to burst correlation
% (Fig S8)
% S8 = Load all recordings

% only plot results for units with at least MIN_SPIKES in recording
MIN_SPIKES = 30;

% specify plot colors
box_colors_all = repmat(["m", "c", "r", "b"], 1,length(av_btb_corr_scores));
box_colors_rand_norm = repmat(["r", "b"], 1,length(av_btb_corr_scores));

% make empty result arrays
vectorized_data_all = [];
boxplot_group_all = [];
vectorized_data_rand_norm = [];
boxplot_group_rand_norm = [];
tick_locs_all = zeros(1,length(av_btb_corr_scores));
tick_locs_rand_norm = zeros(1,length(av_btb_corr_scores));

% for each array
for array = 1:length(av_btb_corr_scores)
    
    % make copy of data
    btb_corr_values = av_btb_corr_scores{array};
    btb_corr_values_rand = av_btb_corr_scores_rand{array};
    
    % set scores for units with less than MIN_SPIKES to NaN
    btb_corr_values(spk_count{array} < MIN_SPIKES) = NaN;
    btb_corr_values_rand(spk_count{array} < MIN_SPIKES) = NaN;
    
    % compute difference between original and shuffled data
    btb_corr_values_diff = (btb_corr_values - btb_corr_values_rand) ./ ...
        (btb_corr_values + btb_corr_values_rand);
    
    % store data for boxplot
    vectorized_data_all = [vectorized_data_all, btb_corr_values(scaf_units{array}), ...
        btb_corr_values(non_scaf_units{array}), btb_corr_values_rand(scaf_units{array}), ...
        btb_corr_values_rand(non_scaf_units{array})];
    vectorized_data_rand_norm = [vectorized_data_rand_norm, btb_corr_values_diff(scaf_units{array}), ...
        btb_corr_values_diff(non_scaf_units{array})];
    
    % store unit type labels
    boxplot_group_all = [boxplot_group_all, (1+(array-1)*4)*ones(1,length(scaf_units{array})), ...
        (2+(array-1)*4)*ones(1,length(non_scaf_units{array})), ...
        (3+(array-1)*4)*ones(1,length(scaf_units{array})), ...
        (4+(array-1)*4)*ones(1,length(non_scaf_units{array}))];
    
    boxplot_group_rand_norm = [boxplot_group_rand_norm, ...
        (1+(array-1)*2)*ones(1,length(scaf_units{array})), ...
        (2+(array-1)*2)*ones(1,length(non_scaf_units{array}))];
    
    tick_locs_all(array) = 2.5+(array-1)*4;
    tick_locs_rand_norm(array) = 1.5+(array-1)*2;
    
end % array

% intiate figure
fig = figure(1);
clf

% adjust size of figure
set(gcf,'PaperPositionMode','auto')
set(fig, 'Position', [100 100 1400 700])
set(fig, 'Renderer', 'painters')

% initiate subplot for boxplots with all data
subplot(2,1,1)

% plot boxplot of consistency scores
bhTemp = boxplot(vectorized_data_all, boxplot_group_all);

% obtain individual boxes
bh = findobj(gca,'Tag','Box');

% color individual boxes
for j=1:length(bh)
    patch(get(bh(j),'XData'),get(bh(j),'YData'),box_colors_all(j),'FaceAlpha',.5);
end

% add y-axis label
ylabel("Av. burst to burst corr.")
ylim([0,1])

% adjust ticks
xticks(tick_locs_all)
xticklabels(rec_names)

% add legend
legend("Non-Rigid (shuffled)", "Backbone (shuffled)", "Non-Rigid", "Backbone", "Location", "SouthOutside", "NumColumns", 2)
legend("boxoff")

set(gca,'linewidth',3)
set(gca, 'fontsize', 14)
set(gca, 'ticklength', [0.003,0.003])
box off


% initiate subplot for boxplots with normalized data
subplot(2,1,2)
hold on

% add marker for 0
yline(0,"k--");

% plot boxplot of consistency scores
boxplot(vectorized_data_rand_norm, boxplot_group_rand_norm);

% obtain individual boxes
bh = findobj(gca,'Tag','Box');

% color individual boxes
for j=1:length(bh)
    patch(get(bh(j),'XData'),get(bh(j),'YData'),box_colors_rand_norm(j),'FaceAlpha',.5);
end

% add y-axis label
ylabel("Av. burst to burst corr. (norm.)")

% adjust ticks
xticks(tick_locs_rand_norm)
xticklabels(rec_names)

set(gca,'linewidth',3)
set(gca, 'fontsize', 14)
set(gca, 'ticklength', [0.003,0.003])
box off


%% compare correlation scores
% (Fig S11)
% S11 = Load all recordings

% only plot results for units with at least MIN_SPIKES in recording
MIN_SPIKES = 30;

% make empty result arrays
sep_labels = [];
tick_locs_all = zeros(1,length(all_pw_corr_vals));
violin_results_all = cell(1,7*length(all_pw_corr_vals));
violin_results_rand_norm = cell(1,4*length(all_pw_corr_vals));

% for each array
for array = 1:length(all_pw_corr_vals)
    
    % make copy of data
    x_corr_vals = all_pw_corr_vals{array};
    x_corr_vals_rand = all_pw_corr_vals_rand{array};
    
    % set diagonal values to NaN
    x_corr_vals(logical(eye(size(x_corr_vals,1)))) = NaN;
    x_corr_vals_rand(logical(eye(size(x_corr_vals,1)))) = NaN;
    
    % set values for units with less than MIN_SPIKES to NaN
    x_corr_vals(spk_count{array} < MIN_SPIKES, :) = NaN;
    x_corr_vals(:, spk_count{array} < MIN_SPIKES) = NaN;
    x_corr_vals_rand(spk_count{array} < MIN_SPIKES, :) = NaN;
    x_corr_vals_rand(:, spk_count{array} < MIN_SPIKES) = NaN;
    
    % compute difference between normal and shuffled data
    x_corr_vals_rand_norm = x_corr_vals - x_corr_vals_rand;
    
    % obtain selection of xcorr values that are correlated
    nz_units = find(sum(x_corr_vals, "omitnan")>0);
    scaf_nz_units = intersect(nz_units,scaf_units{array})';
    non_scaf_nz_units = nz_units;
    non_scaf_nz_units(ismember(non_scaf_nz_units,scaf_nz_units)) = [];
    
    % obtain correlation values per group
    scaf_vs_scaf = x_corr_vals(scaf_nz_units, scaf_nz_units);
    scaf_vs_scaf = scaf_vs_scaf(:);
    scaf_vs_scaf(isnan(scaf_vs_scaf)) = [];
    scaf_vs_scaf_rand = x_corr_vals_rand(scaf_nz_units, scaf_nz_units);
    scaf_vs_scaf_rand = scaf_vs_scaf_rand(:);
    scaf_vs_scaf_rand(isnan(scaf_vs_scaf_rand)) = [];
    scaf_vs_scaf_rand_norm = x_corr_vals_rand_norm(scaf_nz_units, scaf_nz_units);
    scaf_vs_scaf_rand_norm = scaf_vs_scaf_rand_norm(:);
    scaf_vs_scaf_rand_norm(isnan(scaf_vs_scaf_rand_norm)) = [];
    
    
    scaf_vs_non_scaf1 = x_corr_vals(scaf_nz_units, non_scaf_nz_units);
    scaf_vs_non_scaf2 = x_corr_vals(non_scaf_nz_units, scaf_nz_units);
    scaf_vs_non_scaf = [scaf_vs_non_scaf1(:); scaf_vs_non_scaf2(:)];
    scaf_vs_non_scaf(isnan(scaf_vs_non_scaf)) = [];
    scaf_vs_non_scaf1_rand = x_corr_vals_rand(scaf_nz_units, non_scaf_nz_units);
    scaf_vs_non_scaf2_rand = x_corr_vals_rand(non_scaf_nz_units, scaf_nz_units);
    scaf_vs_non_scaf_rand = [scaf_vs_non_scaf1_rand(:); scaf_vs_non_scaf2_rand(:)];
    scaf_vs_non_scaf_rand(isnan(scaf_vs_non_scaf_rand)) = [];
    scaf_vs_non_scaf1_rand_norm = x_corr_vals_rand_norm(scaf_nz_units, non_scaf_nz_units);
    scaf_vs_non_scaf2_rand_norm = x_corr_vals_rand_norm(non_scaf_nz_units, scaf_nz_units);
    scaf_vs_non_scaf_rand_norm = [scaf_vs_non_scaf1_rand_norm(:); scaf_vs_non_scaf2_rand_norm(:)];
    scaf_vs_non_scaf_rand_norm(isnan(scaf_vs_non_scaf_rand_norm)) = [];
    
    
    non_scaf_vs_non_scaf = x_corr_vals(non_scaf_nz_units, non_scaf_nz_units);
    non_scaf_vs_non_scaf = non_scaf_vs_non_scaf(:);
    non_scaf_vs_non_scaf(isnan(non_scaf_vs_non_scaf)) = [];
    non_scaf_vs_non_scaf_rand = x_corr_vals_rand(non_scaf_nz_units, non_scaf_nz_units);
    non_scaf_vs_non_scaf_rand = non_scaf_vs_non_scaf_rand(:);
    non_scaf_vs_non_scaf_rand(isnan(non_scaf_vs_non_scaf_rand)) = [];
    non_scaf_vs_non_scaf_rand_norm = x_corr_vals_rand_norm(non_scaf_nz_units, non_scaf_nz_units);
    non_scaf_vs_non_scaf_rand_norm = non_scaf_vs_non_scaf_rand_norm(:);
    non_scaf_vs_non_scaf_rand_norm(isnan(non_scaf_vs_non_scaf_rand_norm)) = [];
    
    % store data
    violin_results_all{1+(array-1)*7} = scaf_vs_scaf;
    violin_results_all{2+(array-1)*7} = scaf_vs_non_scaf;
    violin_results_all{3+(array-1)*7} = non_scaf_vs_non_scaf;
    violin_results_all{4+(array-1)*7} = scaf_vs_scaf_rand;
    violin_results_all{5+(array-1)*7} = scaf_vs_non_scaf_rand;
    violin_results_all{6+(array-1)*7} = non_scaf_vs_non_scaf_rand;
    violin_results_all{7+(array-1)*7} = -100;
    
    violin_results_rand_norm{1+(array-1)*4} = scaf_vs_scaf_rand_norm;
    violin_results_rand_norm{2+(array-1)*4} = scaf_vs_non_scaf_rand_norm;
    violin_results_rand_norm{3+(array-1)*4} = non_scaf_vs_non_scaf_rand_norm;
    violin_results_rand_norm{4+(array-1)*4} = -100;
    
    % store separate label for white space
    sep_labels = [sep_labels, (4+(array-1)*4)];
    
    % store tick location
    tick_locs_all(array) = 3.5+(array-1)*7;
    tick_locs_rand_norm(array) = 2+(array-1)*4;
    
end % array

% fill empty cells for violin_results_all
for v_cell_all = 1:length(violin_results_all)
    
    if isempty(violin_results_all{v_cell_all})
        
        violin_results_all{v_cell_all} = -100;
        
    end % if
    
end % v_cell

% fill empty cells for violin_results_rand_norm
for v_cell_r = 1:length(violin_results_rand_norm)
    
    if isempty(violin_results_rand_norm{v_cell_r})
        
        violin_results_rand_norm{v_cell_r} = -100;
        
    end % if
    
end % v_cell


% intiate figure
fig = figure(2);
clf

% adjust size of figure
set(gcf,'PaperPositionMode','auto')
set(fig, 'Position', [100 100 1800 700])
set(fig, 'Renderer', 'painters')

% specify violin plot colors
violin_color_mat_all = repmat([0,0,1 ; 0,1,0 ; 1,0,0 ; 0,1,1 ; 1,1,0 ; 1,0,1 ; 1,1,1], length(all_pw_corr_vals), 1);
violin_color_mat_rand_norm = repmat([0,0,1 ; 0,1,0 ; 1,0,0 ; 1,1,1], length(all_pw_corr_vals), 1);

% initiate subplot for violin plot with all data
subplot(2,1,1)

% plot boxplot of consistency scores
violin(violin_results_all, 'mc','r','medc','','plotlegend',0,'facecolor',violin_color_mat_all,'bw',0.05);

% adjust y-axis 
ylim([0,1])
ylabel("Pairwise correlation")

% adjust x-ticks
xticks(tick_locs_all)
xticklabels(rec_names)

set(gca,'linewidth',3)
set(gca, 'fontsize', 14)
set(gca, 'ticklength', [0.003,0.003])
box off


% initiate subplot for violin plot with all data
subplot(2,1,2)
hold on 

% add marker for 0
yline(0,"k--");

% plot boxplot of consistency scores
violin(violin_results_rand_norm, 'mc','r','medc','','plotlegend',0,'facecolor',violin_color_mat_rand_norm,'bw',0.05);

% adjust y-axis 
ylim([-0.5,0.5])
ylabel("Pairwise correlation (norm.)")

% adjust x-ticks
xticks(tick_locs_rand_norm)
xticklabels(rec_names)

set(gca,'linewidth',3)
set(gca, 'fontsize', 14)
set(gca, 'ticklength', [0.003,0.003])
box off
 

%% Compare variance explained by PCA components
% (Fig S14)
% S14 = Load all recordings

% define number of components to include in analysis
NUM_COMP = 3;

% initiate empty result array for grouped bar plot
all_var = NaN(length(var_vals), 3);
all_var_rand = NaN(length(var_vals), 3);

% define labels (hardcoded)
unit_type_labels = [repmat({'a'},1,length(var_vals)), repmat({'s'},1,length(var_vals)), ...
    repmat({'n'},1,length(var_vals))];
model_type_label = repmat([repmat({'org'},1,8), repmat({'sl'},1,6), ...
    repmat({'prim'},1,10)],1,3);

% for each array
for array = 1:length(var_vals)
    
    % store values for first NUM_COMP components
    all_var(array, 1) = sum(var_vals{array}.all(1:NUM_COMP));
    all_var(array, 2) = sum(var_vals{array}.scaff(1:NUM_COMP));
    all_var(array, 3) = sum(var_vals{array}.nscaff(1:NUM_COMP));
    
    all_var_rand(array, 1) = sum(var_vals_rand{array}.all(1:NUM_COMP));
    all_var_rand(array, 2) = sum(var_vals_rand{array}.scaff(1:NUM_COMP));
    all_var_rand(array, 3) = sum(var_vals_rand{array}.nscaff(1:NUM_COMP));
    
    % determine lowest number of components
    min_comp = min([length(var_vals_rand{array}.all), length(var_vals_rand{array}.scaff), ...
        length(var_vals_rand{array}.nscaff)]);
    
    all_var(array, 1) = all_var(array, 1)/sum(var_vals{array}.all(1:min_comp));
    all_var(array, 2) = all_var(array, 2)/sum(var_vals{array}.scaff(1:min_comp));
    all_var(array, 3) = all_var(array, 3)/sum(var_vals{array}.nscaff(1:min_comp));
    all_var_rand(array, 1) = all_var_rand(array, 1)/sum(var_vals_rand{array}.all(1:min_comp));
    all_var_rand(array, 2) = all_var_rand(array, 2)/sum(var_vals_rand{array}.scaff(1:min_comp));
    all_var_rand(array, 3) = all_var_rand(array, 3)/sum(var_vals_rand{array}.nscaff(1:min_comp));
    
end % array

% normalize results with random data
all_var = all_var - all_var_rand;

% compute average and std per model type unit type combination (!!! HARDCODED !!!)
all_mean_org = mean(all_var(1:8,:));
all_std_org = std(all_var(1:8,:));
all_mean_slice = mean(all_var(9:14,:));
all_std_slice = std(all_var(9:14,:));
all_mean_prim = mean(all_var(15:24,:));
all_std_prim = std(all_var(15:24,:));

av_data = [all_mean_org; all_mean_slice; all_mean_prim];
err_data = [all_std_org; all_std_slice; all_std_prim];

% intiate figure
fig = figure(3);
clf

% adjust size of figure
set(gcf,'PaperPositionMode','auto')
set(fig, 'Position', [100 100 2000 400])
set(fig, 'Renderer', 'painters')

% initiate subplot for boxplots with all data
subplot(1,4,[1:3])

% plot grouped barplot of results
bh = bar(all_var, "grouped");
bh(1).FaceColor = 'k';
bh(2).FaceColor = 'b';
bh(3).FaceColor = 'r';
bh(1).FaceAlpha = 0.5;
bh(2).FaceAlpha = 0.5;
bh(3).FaceAlpha = 0.5;

ylabel(sprintf("Norm. frac. first %.0f comp.", NUM_COMP))

xticks([1:length(var_vals)])
xticklabels(rec_names)

set(gca, "FontSize", 14)
set(gca, "LineWidth", 3)
box off

% initiate subplot for category average
subplot(1,4,4)

bh = bar(av_data); % get the bar handles
bh(1).FaceColor = 'k';
bh(2).FaceColor = 'b';
bh(3).FaceColor = 'r';
bh(1).FaceAlpha = 0.5;
bh(2).FaceAlpha = 0.5;
bh(3).FaceAlpha = 0.5;

hold on;

for k = 1:size(av_data,2)
    
    % get x positions per group
    xpos = bh(k).XData + bh(k).XOffset;
    
    % draw errorbar
    errorbar(xpos, av_data(:,k), err_data(:,k), 'LineStyle', 'none', ... 
        'Color', 'k', 'LineWidth', 1);
end

% add axis label
ylabel(sprintf("Norm. frac. first %.0f comp. (av +/- STD)", NUM_COMP))

xticks(1:3)
xticklabels(["Org", "Slice", "Prim"])

set(gca, "FontSize", 14)
set(gca, "LineWidth", 3)
box off


%% store results for statistical analyses (single unit values)

% initiate empty result arrays
unit_numbers = [];
scaf_labels = [];
shuff_labels = [];
sample_labels = [];
firing_rate = [];
btb_corr = [];

% for each array
for array = 1:length(av_rate)
    
    % initiate label arrays
    s_ns_labels = strings(length(av_rate{array}),1);
    rand_labels = strings(2*length(av_rate{array}),1);
    unit_id = strings(length(av_rate{array}),1);
    
    % add scaf non-scaf labels to arrays
    s_ns_labels(scaf_units{array}) = "s";
    s_ns_labels(non_scaf_units{array}) = "n";
    
    % add randomized non-randomized labels to arrays
    rand_labels(1:length(av_rate{array})) = "o";
    rand_labels(length(av_rate{array})+1:end) = "r";
    
    % add organoid names to array
    org_names = repmat(rec_names(array),2*length(av_rate{array}),1);
    
    % for each unit
    for unit = 1:length(av_rate{array})
        
        % store unit id
        unit_id(unit) = rec_names(array) + "_" + num2str(unit);
        
    end % unit
    
    % add to complete array
    unit_numbers = vertcat(unit_numbers, unit_id, unit_id);
    scaf_labels = vertcat(scaf_labels, s_ns_labels, s_ns_labels);
    shuff_labels = vertcat(shuff_labels, rand_labels);
    sample_labels = vertcat(sample_labels, org_names);
    
    % store data in complete array
    firing_rate = vertcat(firing_rate, av_rate{array}', av_rate{array}');
    btb_corr = vertcat(btb_corr, av_btb_corr_scores{array}', av_btb_corr_scores_rand{array}');
    
end % array

% store results in table
stat_data = table(sample_labels, unit_numbers, scaf_labels, shuff_labels, firing_rate, btb_corr);

% store results as csv file
writetable(stat_data,"single_unit_stat_data_org_slice_prim.csv")


%% store results for statistical analyses (pairwise values)

% initiate empty result arrays
unit_numbers = [];
scaf_labels = [];
shuff_labels = [];
sample_labels = [];
pw_corr_vals = [];

% for each array
for array = 1:length(av_rate)
        
    % initiate label arrays
    unit_labels = strings((size(all_pw_corr_vals{array}, 1) * (size(all_pw_corr_vals{array}, 2)-1))/2, 1);
    s_ns_labels = strings((size(all_pw_corr_vals{array}, 1) * (size(all_pw_corr_vals{array}, 2)-1))/2, 1);
    pw_corr = NaN((size(all_pw_corr_vals{array}, 1) * (size(all_pw_corr_vals{array}, 2)-1))/2, 1);
    pw_corr_rand = NaN((size(all_pw_corr_vals{array}, 1) * (size(all_pw_corr_vals{array}, 2)-1))/2, 1);
    
    % set counter to 1
    counter = 1;
    
    % for every row unit
    for row_unit = 1:size(all_pw_corr_vals{array}, 1)
        
        % determine if row unit is a scaf unit or not
        if ismember(row_unit, scaf_units{array})
            scaf_label_row = "s";
        else
            scaf_label_row = "n";
        end % if
        
        % for every column unit
        for col_unit = 1:size(all_pw_corr_vals{array}, 1)
            
            % determine if row unit is a scaf unit or not
            if ismember(col_unit, scaf_units{array})
                scaf_label_col = "s";
            else
                scaf_label_col = "n";
            end % if
        
            % if row unit is different from column unit
            if row_unit < col_unit
                
                % store data in arrays
                unit_labels(counter) = rec_names(array) + "_" + num2str(row_unit) + "_" + num2str(col_unit);
                s_ns_labels(counter) = scaf_label_row + scaf_label_col;
                pw_corr(counter) = all_pw_corr_vals{array}(row_unit, col_unit);
                pw_corr_rand(counter) = all_pw_corr_vals_rand{array}(row_unit, col_unit);

                % add 1 to counter
                counter = counter + 1;
                
            end % if
        end % col_unit
    end % row_unit
        
    % add organoid names to array
    org_names = repmat(rec_names(array),2*(counter-1),1);
    
    % add to complete array
    unit_numbers = vertcat(unit_numbers, unit_labels, unit_labels);
    scaf_labels = vertcat(scaf_labels, s_ns_labels, s_ns_labels);
    shuff_labels = vertcat(shuff_labels, repmat("o", counter-1, 1), repmat("r", counter-1, 1));
    sample_labels = vertcat(sample_labels, org_names);
    
    % store data in complete array
    pw_corr_vals = vertcat(pw_corr_vals, pw_corr, pw_corr_rand);
    
end % array

% store results in table
stat_data = table(sample_labels, unit_numbers, scaf_labels, shuff_labels, pw_corr_vals);

% store results as csv file
writetable(stat_data,"unit_pair_stat_data_org_slice_prim.csv")


%% violin plot function

%__________________________________________________________________________
% violin.m - Simple violin plot using matlab default kernel density estimation
% Last update: 10/2015
%__________________________________________________________________________
% This function creates violin plots based on kernel density estimation
% using ksdensity with default settings. Please be careful when comparing pdfs
% estimated with different bandwidth!
%
% Differently to other boxplot functions, you may specify the x-position.
% This is usefule when overlaying with other data / plots.
%__________________________________________________________________________
%
% Please cite this function as:
% Hoffmann H, 2015: violin.m - Simple violin plot using matlab default kernel
% density estimation. INRES (University of Bonn), Katzenburgweg 5, 53115 Germany.
% hhoffmann@uni-bonn.de
%
%__________________________________________________________________________
%
% INPUT
%
% Y:     Data to be plotted, being either
%        a) n x m matrix. A 'violin' is plotted for each column m, OR
%        b) 1 x m Cellarry with elements being numerical colums of nx1 length.
%
% varargin:
% xlabel:    xlabel. Set either [] or in the form {'txt1','txt2','txt3',...}
% facecolor: FaceColor. (default [1 0.5 0]); Specify abbrev. or m x 3 matrix (e.g. [1 0 0])
% edgecolor: LineColor. (default 'k'); Specify abbrev. (e.g. 'k' for black); set either [],'' or 'none' if the mean should not be plotted
% facealpha: Alpha value (transparency). default: 0.5
% mc:        Color of the bars indicating the mean. (default 'k'); set either [],'' or 'none' if the mean should not be plotted
% medc:      Color of the bars indicating the median. (default 'r'); set either [],'' or 'none' if the median should not be plotted
% bw:        Kernel bandwidth. (default []); prescribe if wanted as follows:
%            a) if bw is a single number, bw will be applied to all
%            columns or cells
%            b) if bw is an array of 1xm or mx1, bw(i) will be applied to cell or column (i).
%            c) if bw is empty (default []), the optimal bandwidth for
%            gaussian kernel is used (see Matlab documentation for
%            ksdensity()
%
% OUTPUT
%
% h:     figure handle
% L:     Legend handle
% MX:    Means of groups
% MED:   Medians of groups
% bw:    bandwidth of kernel
%__________________________________________________________________________
%{
% Example1 (default):
disp('this example uses the statistical toolbox')
Y=[rand(1000,1),gamrnd(1,2,1000,1),normrnd(10,2,1000,1),gamrnd(10,0.1,1000,1)];
[h,L,MX,MED]=violin(Y);
ylabel('\Delta [yesno^{-2}]','FontSize',14)
%Example2 (specify facecolor, edgecolor, xlabel):
disp('this example uses the statistical toolbox')
Y=[rand(1000,1),gamrnd(1,2,1000,1),normrnd(10,2,1000,1),gamrnd(10,0.1,1000,1)];
violin(Y,'xlabel',{'a','b','c','d'},'facecolor',[1 1 0;0 1 0;.3 .3 .3;0 0.3 0.1],'edgecolor','b',...
'bw',0.3,...
'mc','k',...
'medc','r--')
ylabel('\Delta [yesno^{-2}]','FontSize',14)
%Example3 (specify x axis location):
disp('this example uses the statistical toolbox')
Y=[rand(1000,1),gamrnd(1,2,1000,1),normrnd(10,2,1000,1),gamrnd(10,0.1,1000,1)];
violin(Y,'x',[-1 .7 3.4 8.8],'facecolor',[1 1 0;0 1 0;.3 .3 .3;0 0.3 0.1],'edgecolor','none',...
'bw',0.3,'mc','k','medc','r-.')
axis([-2 10 -0.5 20])
ylabel('\Delta [yesno^{-2}]','FontSize',14)
%Example4 (Give data as cells with different n):
disp('this example uses the statistical toolbox')
Y{:,1}=rand(10,1);
Y{:,2}=rand(1000,1);
violin(Y,'facecolor',[1 1 0;0 1 0;.3 .3 .3;0 0.3 0.1],'edgecolor','none','bw',0.1,'mc','k','medc','r-.')
ylabel('\Delta [yesno^{-2}]','FontSize',14)
%}
%%
function[h,L,MX,MED,bw]=violin(Y,varargin)
%defaults:
%_____________________
xL=[];
fc=[1 0.5 0];
lc='k';
alp=0.5;
mc='k';
medc='r';
b=[]; %bandwidth
plotlegend=1;
plotmean=1;
plotmedian=1;
x = [];
%_____________________
%convert single columns to cells:
if iscell(Y)==0
    Y = num2cell(Y,1);
end
%get additional input parameters (varargin)
if isempty(find(strcmp(varargin,'xlabel')))==0
    xL = varargin{find(strcmp(varargin,'xlabel'))+1};
end
if isempty(find(strcmp(varargin,'facecolor')))==0
    fc = varargin{find(strcmp(varargin,'facecolor'))+1};
end
if isempty(find(strcmp(varargin,'edgecolor')))==0
    lc = varargin{find(strcmp(varargin,'edgecolor'))+1};
end
if isempty(find(strcmp(varargin,'facealpha')))==0
    alp = varargin{find(strcmp(varargin,'facealpha'))+1};
end
if isempty(find(strcmp(varargin,'mc')))==0
    if isempty(varargin{find(strcmp(varargin,'mc'))+1})==0
        mc = varargin{find(strcmp(varargin,'mc'))+1};
        plotmean = 1;
    else
        plotmean = 0;
    end
end
if isempty(find(strcmp(varargin,'medc')))==0
    if isempty(varargin{find(strcmp(varargin,'medc'))+1})==0
        medc = varargin{find(strcmp(varargin,'medc'))+1};
        plotmedian = 1;
    else
        plotmedian = 0;
    end
end
if isempty(find(strcmp(varargin,'bw')))==0
    b = varargin{find(strcmp(varargin,'bw'))+1}
    if length(b)==1
        disp(['same bandwidth bw = ',num2str(b),' used for all cols'])
        b=repmat(b,size(Y,2),1);
    elseif length(b)~=size(Y,2)
        warning('length(b)~=size(Y,2)')
        error('please provide only one bandwidth or an array of b with same length as columns in the data set')
    end
end
if isempty(find(strcmp(varargin,'plotlegend')))==0
    plotlegend = varargin{find(strcmp(varargin,'plotlegend'))+1};
end
if isempty(find(strcmp(varargin,'x')))==0
    x = varargin{find(strcmp(varargin,'x'))+1};
end
%%
if size(fc,1)==1
    fc=repmat(fc,size(Y,2),1);
end
%% Calculate the kernel density
i=1;
for i=1:size(Y,2)
    
    if isempty(b)==0
        [f, u, bb]=ksdensity(Y{i},'bandwidth',b(i));
    elseif isempty(b)
        [f, u, bb]=ksdensity(Y{i});
    end
    
    f=f/max(f)*0.3; %normalize
    F(:,i)=f;
    U(:,i)=u;
    MED(:,i)=nanmedian(Y{i});
    MX(:,i)=nanmean(Y{i});
    bw(:,i)=bb;
    
end
%%
%-------------------------------------------------------------------------
% Put the figure automatically on a second monitor
% mp = get(0, 'MonitorPositions');
% set(gcf,'Color','w','Position',[mp(end,1)+50 mp(end,2)+50 800 600])
%-------------------------------------------------------------------------
%Check x-value options
if isempty(x)
    x = zeros(size(Y,2));
    setX = 0;
else
    setX = 1;
    if isempty(xL)==0
        disp('_________________________________________________________________')
        warning('Function is not designed for x-axis specification with string label')
        warning('when providing x, xlabel can be set later anyway')
        error('please provide either x or xlabel. not both.')
    end
end
%% Plot the violins
i=1;
for i=i:size(Y,2)
    if isempty(lc) == 1
        if setX == 0
            h(i)=fill([F(:,i)+i;flipud(i-F(:,i))],[U(:,i);flipud(U(:,i))],fc(i,:),'FaceAlpha',alp,'EdgeColor','none');
        else
            h(i)=fill([F(:,i)+x(i);flipud(x(i)-F(:,i))],[U(:,i);flipud(U(:,i))],fc(i,:),'FaceAlpha',alp,'EdgeColor','none');
        end
    else
        if setX == 0
            h(i)=fill([F(:,i)+i;flipud(i-F(:,i))],[U(:,i);flipud(U(:,i))],fc(i,:),'FaceAlpha',alp,'EdgeColor',lc);
        else
            h(i)=fill([F(:,i)+x(i);flipud(x(i)-F(:,i))],[U(:,i);flipud(U(:,i))],fc(i,:),'FaceAlpha',alp,'EdgeColor',lc);
        end
    end
    hold on
    if setX == 0
        if plotmean == 1
            p(1)=plot([interp1(U(:,i),F(:,i)+i,MX(:,i)), interp1(flipud(U(:,i)),flipud(i-F(:,i)),MX(:,i)) ],[MX(:,i) MX(:,i)],mc,'LineWidth',2);
        end
        if plotmedian == 1
            p(2)=plot([interp1(U(:,i),F(:,i)+i,MED(:,i)), interp1(flipud(U(:,i)),flipud(i-F(:,i)),MED(:,i)) ],[MED(:,i) MED(:,i)],medc,'LineWidth',2);
        end
    elseif setX == 1
        if plotmean == 1
            p(1)=plot([interp1(U(:,i),F(:,i)+i,MX(:,i))+x(i)-i, interp1(flipud(U(:,i)),flipud(i-F(:,i)),MX(:,i))+x(i)-i],[MX(:,i) MX(:,i)],mc,'LineWidth',2);
        end
        if plotmedian == 1
            p(2)=plot([interp1(U(:,i),F(:,i)+i,MED(:,i))+x(i)-i, interp1(flipud(U(:,i)),flipud(i-F(:,i)),MED(:,i))+x(i)-i],[MED(:,i) MED(:,i)],medc,'LineWidth',2);
        end
    end
end
%% Add legend if requested
if plotlegend==1 & plotmean==1 | plotlegend==1 & plotmedian==1
    
    if plotmean==1 & plotmedian==1
        L=legend([p(1) p(2)],'Mean','Median');
    elseif plotmean==0 & plotmedian==1
        L=legend([p(2)],'Median');
    elseif plotmean==1 & plotmedian==0
        L=legend([p(1)],'Mean');
    end
    
    set(L,'box','off','FontSize',14)
else
    L=[];
end
%% Set axis
if setX == 0
    axis([0.5 size(Y,2)+0.5, min(U(:)) max(U(:))]);
elseif setX == 1
    axis([min(x)-0.05*range(x) max(x)+0.05*range(x), min(U(:)) max(U(:))]);
end
%% Set x-labels
xL2={''};
i=1;
for i=1:size(xL,2)
    xL2=[xL2,xL{i},{''}];
end
set(gca,'TickLength',[0 0],'FontSize',12)
box on
if isempty(xL)==0
    set(gca,'XtickLabel',xL2)
end
%-------------------------------------------------------------------------
end %of function
