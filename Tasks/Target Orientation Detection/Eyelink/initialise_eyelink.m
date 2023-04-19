function cfgEyelink = initialise_eyelink(cfgFile, cfgEyelink, cfgScreen)
% cfgEyelink = initialise_eyelink(cfgFile, cfgEyelink, cfgScreen)
% initialise eye link, set parameters and start recording

if cfgEyelink
    try
        if cfgEyelink
            cfgEyelink = el_start(cfgEyelink, cfgScreen, cfgFile);  % set parameters of eyelink and calibrate
        end
    catch
        warning('Eyetracker setup failed! Eyelink triggers will not be sent!');
        while true
            inp2 = input('Do you want to continue? y/n   ','s');
            if inp2 == 'y'
                cfgEyelink = 0;
                break
            elseif inp2 == 'n'
                sca
                error('The experiment aborted by operator.')
            end
        end

    end
end
end