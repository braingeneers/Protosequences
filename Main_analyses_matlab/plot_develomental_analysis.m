%% load data

organoids = ["Or1", "Or2", "Or3" ]; % specify which organoid to analyse
ages = ["6", "7", "8"]; % specify which ages to use

% define path to data
% LOAD_PATH = [path to folder with files]; % !!! adjust path and file names for loading data
        
% make empty cell arrays for loaded results
scaf_units = cell(length(organoids),length(ages));
non_scaf_units = cell(length(organoids),length(ages));
frac_per_unit = cell(length(organoids),length(ages));
av_rate = cell(length(organoids),length(ages));
av_btb_corr_scores = cell(length(organoids),length(ages));
all_pw_corr_vals = cell(length(organoids),length(ages));


% for each date
for age = 1:length(ages)
    
    % for each organoid
    for array = 1:length(organoids)
        
        % load spike data and coordinates
        s_rec_met = load(strcat(LOAD_PATH, sprintf("%s_%sM_single_recording_metrics", organoids(array), ages(age))));
        
        % store results in cell array
        scaf_units{array, age} = s_rec_met.scaf_units;
        non_scaf_units{array, age} = s_rec_met.non_scaf_units;
        frac_per_unit{array, age} = s_rec_met.frac_per_unit;
        av_rate{array, age} = s_rec_met.spk_count./(size(s_rec_met.rate_mat,1)/1000);
        av_btb_corr_scores{array, age} = s_rec_met.av_btb_corr_scores;
        all_pw_corr_vals{array, age} = s_rec_met.all_pw_corr_vals;
        
    end % organoid
end % date


%% Plot fraction of scaffold units at different ages
% (Fig S5A)

% make empty result array
frac_scaf = zeros(length(organoids), length(ages));
leg_lab = strings(1,length(ages));

% for each date
for age = 1:length(ages)
    
    % for each array
    for array = 1:length(organoids)
        
        % compute fraction of scaffold units
        frac_scaf(array, age) = length(scaf_units{array, age})/length(frac_per_unit{array, age});

    end % array
    
    % make legend label
    leg_lab(age) = ages(age) + " months";

end % date


% initiate figure
fig = figure(1);
clf

% adjust size of figure
set(gcf,'PaperPositionMode','auto')
set(fig, 'Position', [10 10 500 350])
set(fig, 'Renderer', 'painters')

bar(frac_scaf)

legend(leg_lab, "Location", "NorthWest", "FontSize", 14)
legend("boxoff")

ylabel("Fraction of backbone units")
xticklabels(organoids)

ax = gca;
ax.FontSize = 14;
ax.LineWidth = 3;
box off


%% plot results of average rate per unit
% (Fig S5B)

plot_scaf_dev_results(av_rate, scaf_units, non_scaf_units, ages, ["r", "b"], "Av. firing rate (Hz)",[0,10])


%% plot results of burst to burst similarity per unit
% (Fig S5C)

plot_scaf_dev_results(av_btb_corr_scores, scaf_units, non_scaf_units, ages, ["r", "b"], "Av. burst corr.",NaN)


%% plot results of average pairwise correlation
% (Fig S5D)

% make cell arrays
av_pair_corr = cell(length(organoids), length(ages));

% for each date
for age = 1:length(ages)
    
    % for each array
    for array = 1:length(organoids)
        
        % compute mean pairwise correlation
        av_pair_corr{array, age} = mean(all_pw_corr_vals{array, age}, "omitnan");
        
    end % array
end % date

plot_scaf_dev_results(av_pair_corr, scaf_units, non_scaf_units, ages, ["r", "b"], "Av. pairwise corr.",NaN)



%% functions

function plot_scaf_dev_results(plot_results, scaf_units, non_scaf_units, date_ages, type_colors, ylab, y_range)

% initiate figure
fig = figure(2);
clf

% adjust size of figure
set(gcf,'PaperPositionMode','auto')
set(fig, 'Position', [10 10 500 350])
set(fig, 'Renderer', 'painters')

% initiate empty result arrays
boxplot_data = [];
boxplot_labels = [];
tick_labels = NaN(1,size(plot_results,1)*2*size(plot_results,2)); % +(size(plot_results,2)-1)
box_colors = char(1,size(plot_results,1)*2*size(plot_results,2)); % +(size(plot_results,2)-1)

% set counter to 1
counter = 1;

% for each array
for array = 1:size(plot_results,1)
    
    % for each date
    for date = 1:size(plot_results,2)

        % select scaffold unit data and labels and store
        boxplot_data = [boxplot_data, plot_results{array,date}(scaf_units{array,date})];
        boxplot_labels = [boxplot_labels, (counter+(array-1))*ones(1,length(plot_results{array,date}(scaf_units{array,date})))];
        
        % store tick label if there was any data
        if ~isempty(plot_results{array,date}(scaf_units{array,date}))
            tick_labels(counter) = date_ages(date);
            box_colors(counter) = type_colors(1);
        else
            box_colors(counter) = 'q';
        end % if
                  
        % select non-scaffold unit data and labels and store
        boxplot_data = [boxplot_data, plot_results{array,date}(non_scaf_units{array,date})];
        boxplot_labels = [boxplot_labels, (counter+size(plot_results,2)+(array-1))*...
            ones(1,length(plot_results{array,date}(non_scaf_units{array,date})))];
        
        % store tick label if there was any data
        if ~isempty(plot_results{array,date}(non_scaf_units{array,date}))
            tick_labels(counter+size(plot_results,2)) = date_ages(date);
            box_colors(counter+size(plot_results,2)) = type_colors(2);
        else
            box_colors(counter+size(plot_results,2)) = 'q';
        end % if
                        
        % add 1 to counter
        counter = counter + 1;
        
    end % date
    
    % add num_dates to counter
    counter = counter + size(plot_results,2);
    
end % array 

% plot boxplot
boxplot(boxplot_data, boxplot_labels, "Labels", ...
    tick_labels(~isnan(tick_labels)), "positions", boxplot_labels);

% remove q labels 
box_colors = box_colors(box_colors~='q');

% obtain individual boxes
bh = findobj(gca,'Tag','Box');

% color individual boxes
for j=1:length(bh)
    patch(get(bh(j),'XData'),get(bh(j),'YData'),box_colors(j),'FaceAlpha',.5);
end

% adjust axis limits
xlim([0,max(boxplot_labels)+1])
if ~isnan(y_range)
    ylim(y_range)
end % if

% add axis lables
ylabel(ylab)

% adjust x-ticks
xticklabels(tick_labels)

ax = gca;
ax.FontSize = 14;
ax.LineWidth = 3;
box off

end % fun plot_scaf_dev_results
