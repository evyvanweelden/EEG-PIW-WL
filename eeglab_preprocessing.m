%% Pre-processing EEG data using EEGlab %%
% By Evy van Weelden and Carl van Beek
% Contact: e.vanweelden@uvt.nl
% Affiliated with Tilburg University
% Last updated on 10-10-2023

% Before cleaning the EEG data, we created an additional row for 
% event codes in the EEG data based on recorded markers.

% Clear workspace and command window before starting
clc
clear

%% Add paths

addpath(genpath('C:/Users/...')); % path to eeglab
addpath(genpath('C:/Users/...')); % path to fieldtrip
addpath(genpath('C:/Users/...')); % add path(s) to data repository
cd 'C:/Users/...' % change current directory to desired folder
clc

%% EEGlab

data_dir = 'C:/Users/...'      % change data directory
processed_dir = 'C:/Users/...' % change directory for processed files

% Initialise EEGLab
[ALLEEG EEG CURRENTSET ALLCOM] = eeglab; % start EEGLAB
pop_editoptions( 'option_storedisk', 0); % Change option to process multiple datasets

% Load one EEG file
EEGfile = data_dir + "name_file.mat";
load(EEGfile)
% %or create a loop over all files in directory using:
% files = {dir(data_dir + "/*.mat").name};
% for i = 1:length(files)
% load(files{i}.name);  % ... continue processing

% Store sampling frequency (fs) and number of channels (nchan)
fs = 250;
nchan = 32;

% Load EEG data into EEGlab structure
eeg_data = EEG.data; % or variable name given to the EEG data
EEG = pop_importdata('dataformat', 'matlab', 'data', eeg_data, 'srate', fs, 'setname', file1);
EEG = pop_chanevent(EEG, 33,'edge','both','edgelen',0); % import events from data channel 33
[ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG);
       
% Set channel locations using location of .loc ile
EEG=pop_chanedit(EEG, 'lookup','C:/Users/.../channellocations32.loc','load',{'C:/Users/.../channellocations32.loc','filetype','autodetect'},'nosedir','+Y');
[ALLEEG EEG] = eeg_store(ALLEEG, EEG, CURRENTSET);       
        
% re-reference
EEG = pop_reref(EEG, [], 'refstate', 0); 
        
% Manual addition of pass band filters through eeglab
EEG = pop_eegfiltnew(EEG, 'locutoff',0.5,'hicutoff',45,'plotfreqz',1);
EEG = eeg_checkset( EEG );

% Manual rejection of artifacts - mark artifacts as green
global rej;
eegplot(EEG.data, 'srate', EEG.srate, 'command', 'global rej,rej=TMPREJ', 'eloc_file', EEG.chanlocs,'winlength', 10, 'events', EEG.event);
uiwait;
tmprej = eegplot2event(rej, -1);
rej_start = tmprej(:, 3);
rej_stop = tmprej(:, 4);
reject = [];
   for k = 1:length(rej_start)
       reject = [reject; rej_start(k) rej_stop(k)];
   end
EEG = eeg_eegrej(EEG, reject); 
[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, CURRENTSET,'setname', 'evdata-filtered-rejected','gui','off');
        
% Prepare EEG struct for intermediate storage
proc_filename = strcat(processed_dir, "/rej/", filename);
EEG.timevec = time;

% Store temporally rejected EEG data
EEG = eeg_checkset(EEG);
save(proc_filename, 'EEG'); 
         
% Run ICA
EEG = pop_runica(EEG, 'icatype', 'runica', 'extended',1,'interrupt','on');
[ALLEEG EEG] = eeg_store(ALLEEG, EEG, CURRENTSET);
EEG = eeg_checkset( EEG );

% Judging ICA components 
EEG = pop_selectcomps(EEG);
pop_plotdata(EEG,0);
[ALLEEG EEG] = eeg_store(ALLEEG, EEG, CURRENTSET);
EEG = eeg_checkset( EEG );
        
% Remove ICA components
uiwait;
EEG = pop_subcomp(EEG);
        
% Save
[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 1,'gui','off');
EEG = eeg_checkset(EEG);
EEG = pop_saveset(EEG, 'filename', 'filtered_rejected_ica_removed', 'filepath', processed_dir);
        
% Prepare EEG struct for final storage
proc_filename = strcat(processed_dir, "/", filename);
EEG.timevec = time;
        
% Store processed EEG data
save(proc_filename, 'EEG');    
close all
