%% load spike data and burst edges

% set paths to all the recordings you want to process
rec_paths = [""]; % add path to where kilosort outputs are saved

% for each recording
for rec = rec_paths  
    
    % load sorted spike data
    ks_spk_data_ont = load(strcat(rec, 't_spk_mat_ks.mat')); % change to appropriate file name
    units = ks_spk_data_ont.units;
    fs = double(ks_spk_data_ont.fs);
    locations = ks_spk_data_ont.locations;
    
    % make empty result arrays
    t_spk_mat = zeros(180000, length(units));
    spike_times = cell(length(units),1);
    xy_raw = zeros(length(units), 2);
    chan_num_sorted = zeros(length(units), 1);
    
    % for each unit
    for unit=1:length(units)
        
        %double check electrode labels are correct
        x = units{1,unit}.x_max;
        y = units{1,unit}.y_max;
        
        % if there are any spikes for unit
        if length(units{1,unit}.spike_train) > 0  
            
            spike_times{unit} = double(units{1,unit}.spike_train)/fs; % spike times (s)
            xy_raw(unit,:) = [x,y]; % coordinates
            
            % for each detected spike
            for spk = 1:length(spike_times{unit})
                
                if int64(spike_times{unit}(spk)*1000) > 0
                    
                    % add 1 to spike matrix
                    t_spk_mat(int64(spike_times{unit}(spk)*1000),unit) = 1;
                    
                end
                
            end % spk
            
        end % if
        
    end % unit

    % save results
    save(strcat(rec, "t_spk_mat_sorted"), "t_spk_mat", "spike_times", "xy_raw", "chan_num_sorted", "-v7.3");

    "variables saved"
        
end % organoid


%% plot template waveform of unit

UNIT = 1;

UNIT_PADDING = 7;
PLOT_PADDING = 150;
ACG_RANGE = 30;

cm = figure(3);
clf

% adjust size of figure
set(gcf,'PaperPositionMode','auto')
set(cm, 'Position', [0 20 1000 300])
set(cm, 'Renderer', 'painters')

hold on

% select averaged waveforms for unit
all_waveforms = units{UNIT}.template;

% % obtain maximum and minimum values per waveform
max_waveform = max(all_waveforms, [], 1);
min_waveform = min(all_waveforms, [], 1);

% select index with largest difference
[~, plot_i] = max(max_waveform - min_waveform);

% %
% initiate subplot
subplot(1,3,1)

% % plot results
plot(linspace(-3,4,size(all_waveforms,1)), 0.195*all_waveforms(:,plot_i), "k")

xlabel("Relative time (ms)")
ylabel("Voltage (uV)")
ax = gca;
ax.FontSize = 14;
xlim([-3,3])


% initiate subplot
subplot(1,3,2)

hold on

% for each waveform
for wf = 1:size(all_waveforms,2)
    
    if wf == plot_i
        color = "r";
    else
        color = "k";
    end % if
    
    % plot waveform
    plot(linspace(-UNIT_PADDING,UNIT_PADDING,size(all_waveforms,1))+locations(wf,1), all_waveforms(:,wf)+locations(wf,2), color)
    
end % wf

% adjust axis
axis equal
xlim([locations(plot_i,1)-PLOT_PADDING, locations(plot_i,1)+PLOT_PADDING])
ylim([locations(plot_i,2)-PLOT_PADDING, locations(plot_i,2)+PLOT_PADDING])

ax = gca;
ax.FontSize = 14;


% initiate subplot
subplot(1,3,3)

% compute ACG
[r,lags] = xcorr(t_spk_mat(:,UNIT), t_spk_mat(:,UNIT), ACG_RANGE);

% remove value at 0 lag
r(lags == 0) = 0;

% plot ACG 
bar(lags,r)

% add axis labels
xlabel("Lag (ms)")
ylabel("Counts")

ax = gca;
ax.FontSize = 14;


