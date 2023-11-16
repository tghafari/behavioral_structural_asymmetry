function cfgEyelink = el_drift_check(cfgEyelink, cfgScreen)
% cfgEyelink = el_start(cfgEyelink, cfgScreen, cfgFile)
% Open screen for calibration, calibrate and start recording

try
    Screen('Preference', 'SkipSyncTests', 1);
    window_EL = Screen('OpenWindow', cfgScreen.scrNum, cfgScreen.backgroundColor, cfgScreen.fullScrn);
    cfgEyelink.defaults = EyelinkInitDefaults(window_EL);  % details about the graphics environment and initializations
    ListenChar(1);  % disable key output to Matlab window - change to 2 on real data collection
    disp('window open')

    disp('Starting Calibration');
    EyelinkDoTrackerSetup(cfgEyelink.defaults);

    disp('Restart Eyetrack recording')
    Screen('Close', window_EL);  % close eyelink screen
    Eyelink('StartRecording');
    WaitSecs(0.1);  % record a few samples before we actually start displaying
    Eyelink('Message', 'SYNCTIME');  % mark zero-plot time in data file
    ListenChar(0);

catch
    warning('error is in el_drift_check')
    cleanup
    psychrethrow(psychlasterror);
end

    function cleanup
        %cleanup routin for Eyelink
        Eyelink('Shutdown');  % shutdown Eyelink
        sca;
        ListenChar(0);  % restore keyboard output to Matlab
    end
end