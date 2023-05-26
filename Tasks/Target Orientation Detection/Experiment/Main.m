% Clear the workspace and the screen
sca;
close all;
clear;

Contrasts_log = -2.5:0.1:0.3;
Contrasts = 10 .^ Contrasts_log;
Attention_Directions = {'Right', 'Left'};
Target_Orientions = {-45, 45};
Distractor_Orientions = {-45, 45};
Repetition_Num = 10;
SF = 9;

Contrast_Num = size(Contrasts,2);
Attention_Direction_Num = size(Attention_Directions,2);
Target_Oriention_Num = size(Target_Orientions,2);
Distractor_Oriention_Num = size(Target_Orientions,2);

Run_Num = Contrast_Num * Attention_Direction_Num * Target_Oriention_Num * ...
    Distractor_Oriention_Num * Repetition_Num;

number_of_short_breaks = 13;
number_of_big_breaks = 3;

Small_Break_Interval = Run_Num / (number_of_short_breaks +1); % 1 Min
Big_Break_Interval = Run_Num / (number_of_big_breaks +1); % 2.5 Min

% Screen properties
PsychDefaultSetup(2);
cfgScreen.scrNum = max(Screen('Screens'));
% get screen number - draw to the external screen if avaliable

[cfgScreen.dispSize.width, cfgScreen.dispSize.height]...
    = Screen('DisplaySize', cfgScreen.scrNum);  % get the physical size of the screen in millimeters
cfgScreen.distance = 50;  % set the distance from participant to the monitor in cm
cfgScreen.resolution = Screen('Resolution', cfgScreen.scrNum);  % get/set the on screen resolution
cfgScreen.fullScrn = [0, 0, cfgScreen.resolution.width, cfgScreen.resolution.height];  % size of full screen in pixels

white = WhiteIndex(cfgScreen.scrNum);
black = BlackIndex(cfgScreen.scrNum);
grey = (white - black) / 2;
cfgScreen.backgroundColor = grey;

Periphery_Pix = angle2pix(cfgScreen,9);
Gabor_Size = angle2pix(cfgScreen,5.6);
Cue_Hight = angle2pix(cfgScreen,1.4);

Cue_Time = 0.6;
Stim_Time = 0.05;
Response_Timeout = 1;

KbName('UnifyKeyNames');
Keyboard.quitKey = KbName('ESCAPE');
Keyboard.confirmKey = KbName('c');

Keyboard.CCWkey = KbName('RightShift'); % CCW +45
Keyboard.CWkey = KbName('LeftShift'); % CW -45

% ------------------------------------------------------------------------
% settings

prompt = {'Subject ID:', 'Session', 'Task', 'Run', 'Eyetracker? y/n', 'Skip sync test? y/n'};
dlgtitle = 'Details';
defaults = {'','01','Orientation_Detection','01','n','n'}; % you can put in default responses
opts.Interpreter = 'tex';
dims = [1, 40; 1, 40; 1, 40; 1, 40; 1, 40; 1, 40];
ansr = inputdlg(prompt, 'Info',dims,defaults,opts); % opens dialog
cfgExp.answer = cell2struct(ansr, {'sub','ses','task','run','eyetracker','skipSync'}, 1);

if strcmp(cfgExp.answer.eyetracker,'y'); cfgEyelink.on = 1; else, cfgEyelink.on = 0; end
if strcmp(cfgExp.answer.skipSync,'y'); skipSync = 1; else, skipSync = 0; end

if(skipSync)

    Screen('Preference', 'SkipSyncTests', 1);

else

    Screen('Preference', 'SkipSyncTests', 0);

end

cfgFile = create_file_directory(cfgExp);  % create file directories

Contrasts = num2cell(Contrasts);
Repetitions = 1:Repetition_Num;
% Necessary For The Proper Working Of BalanceFactors()
Repetitions = num2cell(Repetitions);

States = zeros(Run_Num,1);
States = num2cell(States);

[Run_Contrasts, Run_Attention_Directions, Run_Target_Orientions, Run_Distractor_Orientions, Run_Repetitions] = ...
    BalanceFactors(1, 1, Contrasts, Attention_Directions, Target_Orientions, Distractor_Orientions, Repetitions);

Run_Factors = {Run_Contrasts, Run_Attention_Directions, Run_Target_Orientions, Run_Distractor_Orientions, Run_Repetitions};
Run_Factors = horzcat(Run_Factors{:});

ITIs = zeros(Run_Num,1);

for i = 1:Run_Num

    ITIs(i,1) = (rand/5) + 0.4; % Jitters ITI between 0.4 and 0.6 seconds

end

ITIs = num2cell(ITIs);

ISIs = zeros(Run_Num,1);

for i = 1:Run_Num

    ISIs(i,1) = (rand/5) + 0.5; % Jitters ISI between 0.5 and 0.7 seconds

end

ISIs = num2cell(ISIs);

IDs = zeros(Run_Num,1);

for i = 1:Run_Num

    IDs(i,1) = i;

end

IDs = num2cell(IDs);

Run_Seq = {IDs, States, Run_Factors, ITIs, ISIs, num2cell(zeros(Run_Num,5)), cellstr(strings(Run_Num,1))};
Run_Seq = horzcat(Run_Seq{:});

% Run_Seq : ID, State, Contrast, Attention Direction, Target Oriention,
% Distractor Oriention, Repetition, ITI, ISI, Trial_Onset, Cue_Onset,
% Cue_Offset, Stim_Onset, Stim_Offset, Response Time, Answer

% State :
%
% 1: Done
% 2: -
% 3: No Answer
% 4: Abortion

% ------------------------------------------------------------------------

% Open an on screen window
[window, windowRect] = PsychImaging('OpenWindow', cfgScreen.scrNum, cfgScreen.backgroundColor, cfgScreen.fullScrn);

% Maximum priority
topPriorityLevel = MaxPriority(window);
Priority(topPriorityLevel);

% set up eyelink
cfgEyelink = initialise_eyelink(cfgFile, cfgEyelink, cfgScreen);

% Set up alpha-blending (Global)
Screen('BlendFunction', window, 'GL_SRC_ALPHA', 'GL_ONE_MINUS_SRC_ALPHA');

% Get the size of the on screen window
[screenXpixels, screenYpixels] = Screen('WindowSize', window);

% Get the centre coordinate of the window
[xCenter, yCenter] = RectCenter(windowRect);

%--------------------
% Gabors

Run_Seq_Gabors = cell(Run_Num,1);

% Sigma of Gaussian
sigma = Gabor_Size / 7;

% Parameters
aspectRatio = 1;
phase = 0;

% Build a procedural gabor texture (Note: to get a "standard" Gabor patch
% we set a grey background offset, disable normalisation, and set a
% pre-contrast multiplier of 0.5).
backgroundOffset = [0.5 0.5 0.5 0];
disableNorm = 1;
preContrastMultiplier = 0.5;

gabortex = CreateProceduralGabor(window, Gabor_Size, Gabor_Size, [],...
    backgroundOffset, disableNorm, preContrastMultiplier);

% Spatial Frequency (Cycles Per Pixel)
numCycles = SF;
freq = numCycles / Gabor_Size;

for i = 1:Run_Num

    contrast = Run_Seq{i,3};

    % Properties matrix.
    Run_Seq_Gabors{i,1} = [phase, freq, sigma, contrast, aspectRatio, 0, 0, 0];

end

%--------------------
% Cues

[Right_Cue_Image, ~, alpha] = imread([cfgFile.cue, 'Right.png']);
Right_Cue_Image(:, :, 4) = alpha;

[Left_Cue_Image, ~, alpha] = imread([cfgFile.cue, 'Left.png']);
Left_Cue_Image(:, :, 4) = alpha;

[s1, s2, ~] = size(Right_Cue_Image);

aspect_ratio = s2 / s1;

Cue_Width = Cue_Hight * aspect_ratio;

Cue_Rec = [0 0 Cue_Width Cue_Hight];
Cue_Rec = CenterRectOnPoint(Cue_Rec, xCenter, yCenter);

Right_Cue_Texture = Screen('MakeTexture', window, Right_Cue_Image);
Left_Cue_Texture = Screen('MakeTexture', window, Left_Cue_Image);

%--------------------

% Measure the vertical refresh rate of the monitor
ifi = Screen('GetFlipInterval', window);

Cue_Frames = round(Cue_Time / ifi);
Stim_Frames = round(Stim_Time / ifi);
Response_Timeout_Frames = round(Response_Timeout / ifi);

% Setup the text type for the window
Screen('TextFont', window, 'Ariel');
Screen('TextSize', window, 32);

% Size of the fixation cross
FixCross_DimPix = 16;

FixCross_xCoords = [-FixCross_DimPix FixCross_DimPix 0 0];
FixCross_yCoords = [0 0 -FixCross_DimPix FixCross_DimPix];
FixCross_allCoords = [FixCross_xCoords; FixCross_yCoords];

% Set the line width for the fixation cross
FixCross_lineWidthPix = 3;

% Set the line width for the fixation circle
FixCircle_lineWidthPix = 4;

% Making a base Rect
FixCircle_baseRect = [-FixCross_DimPix -FixCross_DimPix FixCross_DimPix FixCross_DimPix];

% Center the rectangle on the centre of the screen
FixCircle_centeredRect = CenterRectOnPoint(FixCircle_baseRect, xCenter, yCenter);

DrawFormattedText(window, 'Press Anykey To Start :)', 'center', 'center',[1 1 1]);

% Flip to the screen
Screen('Flip', window);

% Wait for a key press
KbStrokeWait;

ListenChar(-1); % makes it so characters typed dont show up in the command window
HideCursor(); % hides the cursor

Keyboard.activeKeys = [Keyboard.quitKey, Keyboard.confirmKey, Keyboard.CCWkey, Keyboard.CWkey];
Keyboard.responseKeys = [Keyboard.CCWkey, Keyboard.CWkey];
Keyboard.deviceNum = -1;  % listen to all devices during test/train

% only listen for specific keys
scanList = zeros(1,256);
scanList(Keyboard.activeKeys) = 1;
KbQueueCreate(Keyboard.deviceNum,scanList);  % create queue
KbQueueStart;  % start listening to input
KbQueueFlush;  % clear all keyboard presses so far

% Main Task ///////////////////////////////////////////////////////////

Abortion = 0;
Abortion_Pauses = zeros(Run_Num,1);

Task_Onset = GetSecs();
send_trigger(cfgEyelink, 'Task Onset');

for n = 1:Run_Num

    if (Abortion == 1)

        send_trigger(cfgEyelink, 'Abortion');
        break;

    end

    if ((ceil(n / Big_Break_Interval) ~= ceil((n-1) / Big_Break_Interval)) && n ~= 1)

        DrawFormattedText(window, 'Break For 2.5 Min :)', 'center', 'center',[1 1 1]);
        vbl=Screen('Flip',window); % swaps backbuffer to frontbuffer

        DrawFormattedText(window, 'Press Anykey To Start :)', 'center', 'center',[1 1 1]);

        Screen('Flip',window,vbl + 150);

        if cfgEyelink.on
            el_drift_check(cfgEyelink, cfgScreen);
        end

        % Wait for a key press
        KbStrokeWait;

    elseif ((ceil(n / Small_Break_Interval) ~= ceil((n-1) / Small_Break_Interval)) && n ~= 1)

        DrawFormattedText(window, 'Break For 1 Min :)', 'center', 'center',[1 1 1]);
        vbl=Screen('Flip',window); % swaps backbuffer to frontbuffer

        DrawFormattedText(window, 'Press Anykey To Start :)', 'center', 'center',[1 1 1]);

        Screen('Flip',window,vbl + 60);

        % Wait for a key press
        KbStrokeWait;

    end

    if (strcmp(Run_Seq{n,4}, 'Right'))
        Target_Gabor_Position = [xCenter + Periphery_Pix, yCenter];
        Distractor_Gabor_Position = [xCenter - Periphery_Pix, yCenter];

    elseif (strcmp(Run_Seq{n,4}, 'Left'))
        Target_Gabor_Position = [xCenter - Periphery_Pix, yCenter];
        Distractor_Gabor_Position = [xCenter + Periphery_Pix, yCenter];

    end

    Gabor_Rec = [0 0 Gabor_Size Gabor_Size];

    Target_Gabor_Rec = CenterRectOnPoint(Gabor_Rec, Target_Gabor_Position(1), Target_Gabor_Position(2));
    Distractor_Gabor_Rec = CenterRectOnPoint(Gabor_Rec, Distractor_Gabor_Position(1), Distractor_Gabor_Position(2));

    % ITI

    ITI_Frames = round(Run_Seq{n,8} / ifi);

    for frame = 1:ITI_Frames

        % Draw the fixation cross
        Screen('DrawLines', window, FixCross_allCoords, FixCross_lineWidthPix, black, [xCenter yCenter], 2);

        if (frame == 1)
            Trial_Onset = Screen('Flip',window); % swaps backbuffer to frontbuffer
            vbl = Trial_Onset;
            send_trigger(cfgEyelink, 'Trial Onset');
        else
            vbl = Screen('Flip', window, vbl + (0.5 * ifi));
        end

    end

    % Cue

    for frame = 1:Cue_Frames

        if (strcmp(Run_Seq{n,4}, 'Right'))

            Screen('DrawTexture', window, Right_Cue_Texture, [], Cue_Rec);
            if (frame == 1)
                send_trigger(cfgEyelink, 'Right Cue');
            end

        elseif (strcmp(Run_Seq{n,4}, 'Left'))

            Screen('DrawTexture', window, Left_Cue_Texture, [], Cue_Rec);
            if (frame == 1)
                send_trigger(cfgEyelink, 'Left Cue');
            end

        end

        if (frame == 1)
            Cue_Onset = Screen('Flip',window); % swaps backbuffer to frontbuffer
            vbl = Cue_Onset;
        else
            vbl = Screen('Flip', window, vbl + (0.5 * ifi));
        end

    end

    % ISI

    ISI_Frames = round(Run_Seq{n,9} / ifi);

    for frame = 1:ISI_Frames

        % Draw the fixation cross
        Screen('DrawLines', window, FixCross_allCoords, FixCross_lineWidthPix, black, [xCenter yCenter], 2);

        if (frame == 1)
            Cue_Offset = Screen('Flip',window); % swaps backbuffer to frontbuffer
            vbl = Cue_Offset;
        else
            vbl = Screen('Flip', window, vbl + (0.5 * ifi));
        end

    end

    % Stim (Gabors)
    for frame = 1:Stim_Frames

        % Draw the fixation cross
        Screen('DrawLines', window, FixCross_allCoords, FixCross_lineWidthPix, black, [xCenter yCenter], 2);

        % Draw the fixation circle
        Screen('FrameOval', window, black, FixCircle_centeredRect, FixCircle_lineWidthPix);

        % Disable alpha-blending for Gabors
        Screen('BlendFunction', window, 'GL_ONE', 'GL_ZERO');

        % Draw the Gabors
        Screen('DrawTexture', window, gabortex, [], Target_Gabor_Rec, ...
            Run_Seq{n,5}, [], [], [], [], kPsychDontDoRotation, Run_Seq_Gabors{n,1}');

        Screen('DrawTexture', window, gabortex, [], Distractor_Gabor_Rec, ...
            Run_Seq{n,6}, [], [], [], [], kPsychDontDoRotation, Run_Seq_Gabors{n,1}');

        % Set up alpha-blending (Global)
        Screen('BlendFunction', window, 'GL_SRC_ALPHA', 'GL_ONE_MINUS_SRC_ALPHA');

        if (frame == 1)
            Stim_Onset = Screen('Flip',window); % swaps backbuffer to frontbuffer
            vbl = Stim_Onset;
            send_trigger(cfgEyelink, 'Stim Onset');
            KbQueueFlush; % Flushes Buffer
        else
            vbl = Screen('Flip', window, vbl + (0.5 * ifi));
        end

    end

    % Draw the fixation cross
    Screen('DrawLines', window, FixCross_allCoords, FixCross_lineWidthPix, black, [xCenter yCenter], 2);

    Stim_Offset = Screen('Flip',window);
    send_trigger(cfgEyelink, 'Stim Offset');

    noResp = 1;

    while (noResp == 1)

        [presd, firstPrsd] = KbQueueCheck;  % listens for response
        keyCod = find(firstPrsd,1);  % collects the pressed key code

        if (presd && (ismember(keyCod,Keyboard.responseKeys))) % store response variables

            send_trigger(cfgEyelink, 'Response Onset');
            Response_Key_Time = firstPrsd(keyCod);  % exact time of button press
            Key = KbName(keyCod);  % which key was pressed
            Key = string(Key);

            Run_Seq{n,15} = Response_Key_Time;
            Run_Seq{n,16} = Key;
            Run_Seq{n,2} = 1; % 1: Done

            noResp = 0;
            break;

        elseif (presd && keyCod == Keyboard.quitKey)

            warning('Experiment aborted!')
            Abortion_Pauses(n,1) = Abortion_Pauses(n,1) + 1;
            send_trigger(cfgEyelink, 'Abortion Pause');

            DrawFormattedText(window, 'Press C to confirm exit or any other key to continue', 'center', 'center',[1 1 1]);
            Screen('Flip',window);

            [~, abrtPrsd] = KbStrokeWait;
            if abrtPrsd(Keyboard.confirmKey)  % Press Any Other Key To Resume

                Abortion = 1;
                Run_Seq{n,2} = 4; % 4: Abortion
                Run_Seq{n,15} = NaN;
                Run_Seq{n,16} = 'None' ;

                noResp = 0;
                break;

            end

            % -----------------------------------------
            % Repeating Trial

            send_trigger(cfgEyelink, 'Repeating Trial');

            % ITI

            for frame = 1:ITI_Frames

                % Draw the fixation cross
                Screen('DrawLines', window, FixCross_allCoords, FixCross_lineWidthPix, black, [xCenter yCenter], 2);

                if (frame == 1)
                    Trial_Onset = Screen('Flip',window); % swaps backbuffer to frontbuffer
                    vbl = Trial_Onset;
                    send_trigger(cfgEyelink, 'Trial Onset');
                else
                    vbl = Screen('Flip', window, vbl + (0.5 * ifi));
                end

            end

            % Cue

            for frame = 1:Cue_Frames

                if (strcmp(Run_Seq{n,4}, 'Right'))

                    Screen('DrawTexture', window, Right_Cue_Texture, [], Cue_Rec);
                    if (frame == 1)
                        send_trigger(cfgEyelink, 'Right Cue');
                    end

                elseif (strcmp(Run_Seq{n,4}, 'Left'))

                    Screen('DrawTexture', window, Left_Cue_Texture, [], Cue_Rec);
                    if (frame == 1)
                        send_trigger(cfgEyelink, 'Left Cue');
                    end

                end

                if (frame == 1)
                    Cue_Onset = Screen('Flip',window); % swaps backbuffer to frontbuffer
                    vbl = Cue_Onset;
                else
                    vbl = Screen('Flip', window, vbl + (0.5 * ifi));
                end

            end

            % ISI

            for frame = 1:ISI_Frames

                % Draw the fixation cross
                Screen('DrawLines', window, FixCross_allCoords, FixCross_lineWidthPix, black, [xCenter yCenter], 2);

                if (frame == 1)
                    Cue_Offset = Screen('Flip',window); % swaps backbuffer to frontbuffer
                    vbl = Cue_Offset;
                else
                    vbl = Screen('Flip', window, vbl + (0.5 * ifi));
                end

            end

            % Stim (Gabors)
            for frame = 1:Stim_Frames

                % Draw the fixation cross
                Screen('DrawLines', window, FixCross_allCoords, FixCross_lineWidthPix, black, [xCenter yCenter], 2);

                % Draw the fixation circle
                Screen('FrameOval', window, black, FixCircle_centeredRect, FixCircle_lineWidthPix);

                % Disable alpha-blending for Gabors
                Screen('BlendFunction', window, 'GL_ONE', 'GL_ZERO');

                % Draw the Gabors
                Screen('DrawTexture', window, gabortex, [], Target_Gabor_Rec, ...
                    Run_Seq{n,5}, [], [], [], [], kPsychDontDoRotation, Run_Seq_Gabors{n,1}');

                Screen('DrawTexture', window, gabortex, [], Distractor_Gabor_Rec, ...
                    Run_Seq{n,6}, [], [], [], [], kPsychDontDoRotation, Run_Seq_Gabors{n,1}');

                % Set up alpha-blending (Global)
                Screen('BlendFunction', window, 'GL_SRC_ALPHA', 'GL_ONE_MINUS_SRC_ALPHA');

                if (frame == 1)
                    Stim_Onset = Screen('Flip',window); % swaps backbuffer to frontbuffer
                    vbl = Stim_Onset;
                    send_trigger(cfgEyelink, 'Stim Onset');
                    KbQueueFlush; % Flushes Buffer
                else
                    vbl = Screen('Flip', window, vbl + (0.5 * ifi));
                end

            end

            % Draw the fixation cross
            Screen('DrawLines', window, FixCross_allCoords, FixCross_lineWidthPix, black, [xCenter yCenter], 2);

            Stim_Offset = Screen('Flip',window);
            send_trigger(cfgEyelink, 'Stim Offset');

            % -----------------------------------------

        elseif ((GetSecs - Stim_Offset) > Response_Timeout)  % Stop listening

            Run_Seq{n,2} = 3; % 3: No Answer
            Run_Seq{n,15} = NaN;
            Run_Seq{n,16} = 'None' ;

            noResp = 0;
            break;

        end

    end

    Run_Seq{n,10} = Trial_Onset;
    Run_Seq{n,11} = Cue_Onset;
    Run_Seq{n,12} = Cue_Offset;
    Run_Seq{n,13} = Stim_Onset;
    Run_Seq{n,14} = Stim_Offset;

end

Task_Offset = send_trigger(cfgEyelink, 'End of experiment');

DrawFormattedText(window, 'Press anykey to exit :)', 'center', 'center',[1 1 1]);

Screen('Flip',window); % swaps backbuffer to frontbuffer

% Wait for a key press
KbStrokeWait;

ListenChar(0); % Makes it so characters typed do show up in the command window
ShowCursor(); % Shows the cursor
Screen('CloseAll'); % Closes Screen

% Clear the screen
sca;

%% saving and cleaning up

cfgOutput.Output_table = cell2table(Run_Seq,"VariableNames",["ID", "State", "Contrast", ...
    "Attention_Direction", "Target_Oriention", "Distractor_Oriention", ...
    "Repetition", "ITI", "ISI", "Trial_Onset", "Cue_Onset", ...
    "Cue_Offset", "Stim_Onset", "Stim_Offset", "Response_Time", "Answer"]);

% check if the logfile is being overwritten
if exist([cfgFile.subDir, cfgFile.BIDSname, cfgFile.logFile], 'file') > 0
    warning('log file will be overwritten!');
    cont = input('Do you want to continue? (y/n) ','s');
    while true
        if cont == 'y'
            break
        elseif cont == 'n'
            error('The experiment aborted by operator.')
        end
    end
end

try
    save([cfgFile.subDir, cfgFile.BIDSname, cfgFile.logFile])
    writetable(cfgOutput.Output_table,[cfgFile.subDir, cfgFile.BIDSname, cfgFile.csvFile]);
catch
    warning('Saving the log files failed.');
end

try
    if cfgEyelink.on
        el_stop(cfgFile)
    end
catch
    warning('Stopping the Eyelink failed');
end

Priority(0);