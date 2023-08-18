function Staircase_Results = check_staircase(cfgExp)

tmp = matlab.desktop.editor.getActive;
cd(fileparts(tmp.Filename));  % move to the current directory
cd ..  % move up one folder

addpath([cd, '\Functions\']);  % add the sub-functions
addpath([cd, '\Results\']);  % add result folder

Staircase_Check.res = [cd, '\Results\'] ;

subject_ID = cfgExp.answer.sub; % makes subject ID

Staircase_Check.subDir = [Staircase_Check.res, 'sub-', subject_ID, filesep, ...
    'ses-' cfgExp.answer.staircase_ses, filesep, 'beh', filesep];  % store subject directory address

cfgExp.answer.task = 'Orientation_Detection_Staircase';

Staircase_Check.BIDSname = ['sub-', subject_ID, '_', 'ses-', cfgExp.answer.staircase_ses, '_'...
    , 'task-', cfgExp.answer.task, '_', 'run-', cfgExp.answer.staircase_run];  % BIDS specific file name

Staircase_Check.logFile = ['_logfile', '.mat'];  % logfile file name

try

    load([Staircase_Check.subDir, Staircase_Check.BIDSname, Staircase_Check.logFile], "cfgOutput");
    Staircase_Results.Contrast_Threshold = cfgOutput.Contrast_Threshold;
    Staircase_Results.State = 1;

catch

    Staircase_Results.State = 0;

end

end