function cfgFile = create_file_directory(cfgExp)
% cfgFile = create_file_directory(cfgExp)
% cd to and creates subject directory according to OS

tmp = matlab.desktop.editor.getActive;
cd(fileparts(tmp.Filename));  % move to the current directory
cd ..  % move up one folder

addpath([cd, '\Functions\']);  % add the sub-functions
addpath([cd, '\Eyelink\']);  % add Eyelink functions
addpath([cd, '\Cue\']);  % add stimulus images
addpath([cd, '\Results\']);  % add result folder

cfgFile.res = [cd, '\Results\'] ;
cfgFile.cue = [cd, '\Cue\'];

% clock_info = clock; % Current date and time as date vector. [year month day hour minute seconds]
subject_ID = cfgExp.answer.sub; % makes subject ID

mkdir([cfgFile.res, 'sub-', subject_ID, filesep, 'ses-', cfgExp.answer.ses, filesep, 'beh', filesep]);  % make result directory with BIDS format
cfgFile.subDir = [cfgFile.res, 'sub-', subject_ID, filesep, 'ses-' cfgExp.answer.ses, filesep, 'beh', filesep];  % store subject directory address
cfgFile.BIDSname = ['sub-', subject_ID, '_', 'ses-', cfgExp.answer.ses, '_'...
    , 'task-', cfgExp.answer.task, '_', 'run-', cfgExp.answer.run];  % BIDS specific file name

cfgFile.edfFile = ['_eyetracking', '.edf'];  % eyetracking file name
cfgFile.eyelink = ['e', cfgExp.answer.run, subject_ID];  % file name to use on eyelink pc
cfgFile.logFile = ['_logfile', '.mat'];  % logfile file name
cfgFile.csvFile = ['_logfile', '.csv'];  % logfile file name

end