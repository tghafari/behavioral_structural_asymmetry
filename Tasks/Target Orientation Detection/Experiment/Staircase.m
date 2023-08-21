% Clear the workspace and the screen
sca;
close all;
clear;

Target_Orientions = {-45, 45};
Repetition_Num = 4;
SF = 9;

Straircase_Step_Num = 16;
Block_Repetition_Num = 1; % Number of Pair Right/Left Blocks
Initial_Contrast = 1;

Block_Num = 2 * Block_Repetition_Num;
Target_Oriention_Num = size(Target_Orientions,2);

Straircase_Step_Run_Num = Target_Oriention_Num * Repetition_Num;
Block_Run_Num = Straircase_Step_Run_Num * Straircase_Step_Num;
Run_Num = Block_Run_Num * Block_Num;

number_of_short_breaks = 1;
number_of_big_breaks = 1;

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

Cue_Time = 0.2;
Stim_Time = 0.05;
Response_Timeout = 1;

KbName('UnifyKeyNames');
Keyboard.quitKey = KbName('ESCAPE');
Keyboard.confirmKey = KbName('c');

Keyboard.CCWkey = KbName('RightShift'); % CCW +45
Keyboard.CWkey = KbName('LeftShift'); % CW -45

% ------------------------------------------------------------------------

prompt = {'Subject ID:', 'Session', 'Task \color{Red}(Staircase)', 'Run', 'Eyetracker? y/n', 'Skip sync test? y/n', 'First Block Question?'};
defaults = {'','01','Orientation_Detection_Staircase','01','n','n', 'Right'}; % you can put in default responses
opts.Interpreter = 'tex';
dims = [1, 40; 1, 40; 1, 40; 1, 40; 1, 40; 1, 40; 1, 40];
ansr = inputdlg(prompt, 'Info', dims, defaults, opts); % opens dialog
cfgExp.answer = cell2struct(ansr, {'sub','ses','task','run','eyetracker','skipSync', 'First_Block_Question'}, 1);

if strcmp(cfgExp.answer.eyetracker,'y'); cfgEyelink.on = 1; else, cfgEyelink.on = 0; end
if strcmp(cfgExp.answer.skipSync,'y'); skipSync = 1; else, skipSync = 0; end
if strcmp(cfgExp.answer.First_Block_Question,'Right'); First_Block_Question = 1; else, First_Block_Question = 0; end

if(skipSync)

    Screen('Preference', 'SkipSyncTests', 1);

else

    Screen('Preference', 'SkipSyncTests', 0);

end

cfgFile = create_file_directory(cfgExp);  % create file directories

Repetitions = 1:Repetition_Num;
% Necessary For The Proper Working Of BalanceFactors()
Repetitions = num2cell(Repetitions);

IDs = 1:Run_Num;
IDs = num2cell(IDs');
States = zeros(Run_Num,1);
States = num2cell(States);

Blocks = 1:Block_Num;
Run_Blocks = repelem(Blocks,Block_Run_Num);
Run_Blocks = num2cell(Run_Blocks');

Run_Block_Questions = cell(Run_Num,1);

for i = 1:Run_Num

    if (mod(Run_Blocks{i,1},2) == First_Block_Question)

        Run_Block_Questions{i,1} = "Right";

    else

        Run_Block_Questions{i,1} = "Left";

    end

end

Straircase_Steps = 1:1:Straircase_Step_Num;
Block_Straircase_Steps = repelem(Straircase_Steps,Straircase_Step_Run_Num);
Run_Straircase_Steps = repmat(Block_Straircase_Steps,1,Block_Num);
Run_Straircase_Steps = num2cell(Run_Straircase_Steps');

Run_Factors = cell(0);

for i = 1:Run_Num

    if (i == Run_Num)

        [Run_Target_Orientions, Run_Repetitions] = ...
            BalanceFactors(1, 1, Target_Orientions, Repetitions);

        Run_Contrasts = cell(Straircase_Step_Run_Num,1);

        if (Run_Straircase_Steps{i,1} == 1)

            Run_Contrasts = num2cell(repelem(Initial_Contrast,Straircase_Step_Run_Num)');

        end

        Straircase_Step_Run_Factors = {Run_Target_Orientions, Run_Repetitions, Run_Contrasts};
        Straircase_Step_Run_Factors = horzcat(Straircase_Step_Run_Factors{:});

        Run_Factors = {Run_Factors, Straircase_Step_Run_Factors};
        Run_Factors = vertcat(Run_Factors{:});

    elseif ((Run_Straircase_Steps{i,1} ~= Run_Straircase_Steps{i+1,1}))

        [Run_Target_Orientions, Run_Repetitions] = ...
            BalanceFactors(1, 1, Target_Orientions, Repetitions);

        Run_Contrasts = cell(Straircase_Step_Run_Num,1);

        if (Run_Straircase_Steps{i,1} == 1)

            Run_Contrasts = num2cell(repelem(Initial_Contrast,Straircase_Step_Run_Num)');

        end

        Straircase_Step_Run_Factors = {Run_Target_Orientions, Run_Repetitions, Run_Contrasts};
        Straircase_Step_Run_Factors = horzcat(Straircase_Step_Run_Factors{:});

        Run_Factors = {Run_Factors, Straircase_Step_Run_Factors};
        Run_Factors = vertcat(Run_Factors{:});

    end

end

ITIs = zeros(Run_Num,1);

for i = 1:Run_Num

    ITIs(i,1) = (rand/5) + 0.4; % Jitters ITI between 0.4 and 0.6 seconds

end

ITIs = num2cell(ITIs);

ISIs = zeros(Run_Num,1);

for i = 1:Run_Num

    ISIs(i,1) = (rand/5) + 0.9; % Jitters ISI between 0.9 and 1.1 seconds

end

ISIs = num2cell(ISIs);

Run_Seq = {IDs, States, Run_Blocks, Run_Block_Questions, Run_Straircase_Steps,...
    Run_Factors, ITIs, ISIs, num2cell(zeros(Run_Num,5)),...
    cellstr(strings(Run_Num,1))};

Run_Seq = horzcat(Run_Seq{:});

% Run_Seq : ID, State, Block Number, Block Question (Attention Direction),
% Straircase Step, Target Oriention, Repetition, Contrast, ITI, ISI,
% Trial_Onset, Cue_Onset, Cue_Offset, Stim_Onset, Stim_Offset,
% Response Time, Answer

% State :
%
% 1: Done
% 2: -
% 3: No Answer
% 4: Abortion

% Staircase_Results : Block Number, Block Question (Attention Direction),
% Contrast Threshold

Staircase_Results = {num2cell(Blocks.'), cellstr(strings(Block_Num,1)), num2cell(zeros(Block_Num,1))};
Staircase_Results = horzcat(Staircase_Results{:});

for i = 1:Block_Num

    if (mod(Staircase_Results{i,1},2) == First_Block_Question)

        Staircase_Results{i,2} = "Right";

    else

        Staircase_Results{i,2} = "Left";

    end

end

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

    % Break Check -----------------------------------------------------

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

    % Block Check -----------------------------------------------------

    if (n == 1)

        if (strcmp(Run_Seq{n,4}, 'Right'))

            DrawFormattedText(window, 'Next Block: Right', 'center', 'center',[1 1 1]);

        elseif (strcmp(Run_Seq{n,4}, 'Left'))

            DrawFormattedText(window, 'Next Block: Left', 'center', 'center',[1 1 1]);

        end

        vbl=Screen('Flip',window); % swaps backbuffer to frontbuffer

        DrawFormattedText(window, 'Press Anykey To Start :)', 'center', 'center',[1 1 1]);

        Screen('Flip',window,vbl + 4);

        % Wait for a key press
        KbStrokeWait;

    elseif (Run_Seq{n,3} ~= Run_Seq{n-1,3})

        if (strcmp(Run_Seq{n,4}, 'Right'))

            DrawFormattedText(window, 'Next Block: Right', 'center', 'center',[1 1 1]);

        elseif (strcmp(Run_Seq{n,4}, 'Left'))

            DrawFormattedText(window, 'Next Block: Left', 'center', 'center',[1 1 1]);

        end

        vbl=Screen('Flip',window); % swaps backbuffer to frontbuffer

        DrawFormattedText(window, 'Press Anykey To Start :)', 'center', 'center',[1 1 1]);

        Screen('Flip',window,vbl + 4);

        % Wait for a key press
        KbStrokeWait;

    end

    % Staircase Processing -------------------------------

    if (isempty(Run_Seq{n,8}))

        % Processing_Run_Seq : Target Oriention, Answer, Contrast

        Processing_Run_Seq = Run_Seq(n-Straircase_Step_Run_Num:n-1,[6 17 8]);

        Processing_Count = 0;

        for j = 1:Straircase_Step_Run_Num

            if(Processing_Run_Seq{j,1} == 45) && ...
                    (strcmp(Processing_Run_Seq{j,2}, 'LeftShift'))

                Processing_Count = Processing_Count +1;

            elseif(Processing_Run_Seq{j,1} == -45) && ...
                    (strcmp(Processing_Run_Seq{j,2}, 'RightShift'))

                Processing_Count = Processing_Count +1;

            end

        end

        Current_Contrast = Processing_Run_Seq{j,3};

        if(Processing_Count >= (Straircase_Step_Run_Num * 0.8))

            for k = n:(n+Straircase_Step_Run_Num-1)

                Run_Seq{k,8} = 0.5 * Current_Contrast;

            end

        elseif(Processing_Count < (Straircase_Step_Run_Num * 0.7))

            for k = n:(n+Straircase_Step_Run_Num-1)

                Run_Seq{k,8} = 2 * Current_Contrast;

                if(Run_Seq{k,8}> Initial_Contrast)

                    Run_Seq{k,8} = Initial_Contrast;

                end

            end

        else

            for k = n:(n+Straircase_Step_Run_Num-1)

                Run_Seq{k,8} = Current_Contrast;

            end

        end

    end

    % ------------------------------------------------------

    contrast = Run_Seq{n,8};

    % Properties matrix.
    Run_Seq_Gabor = [phase, freq, sigma, contrast, aspectRatio, 0, 0, 0];

    if (strcmp(Run_Seq{n,4}, 'Right'))
        Target_Gabor_Position = [xCenter + Periphery_Pix, yCenter];

    elseif (strcmp(Run_Seq{n,4}, 'Left'))
        Target_Gabor_Position = [xCenter - Periphery_Pix, yCenter];

    end

    Gabor_Rec = [0 0 Gabor_Size Gabor_Size];

    Target_Gabor_Rec = CenterRectOnPoint(Gabor_Rec, Target_Gabor_Position(1), Target_Gabor_Position(2));

    % ITI

    ITI_Frames = round(Run_Seq{n,9} / ifi);

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

    ISI_Frames = round(Run_Seq{n,10} / ifi);

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
            Run_Seq{n,6}, [], [], [], [], kPsychDontDoRotation, Run_Seq_Gabor');

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

            Run_Seq{n,16} = Response_Key_Time;
            Run_Seq{n,17} = Key;
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
                Run_Seq{n,16} = NaN;
                Run_Seq{n,17} = 'None' ;

                noResp = 0;
                break;

            end

            % -----------------------------------------
            % Repeating Trial

            send_trigger(cfgEyelink, 'Repeating Trial');

            % ITI

            ITI_Frames = round(Run_Seq{n,9} / ifi);

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

            ISI_Frames = round(Run_Seq{n,10} / ifi);

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
                    Run_Seq{n,6}, [], [], [], [], kPsychDontDoRotation, Run_Seq_Gabor');

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
            Run_Seq{n,16} = NaN;
            Run_Seq{n,17} = 'None' ;

            noResp = 0;
            break;

        end

    end

    Run_Seq{n,11} = Trial_Onset;
    Run_Seq{n,12} = Cue_Onset;
    Run_Seq{n,13} = Cue_Offset;
    Run_Seq{n,14} = Stim_Onset;
    Run_Seq{n,15} = Stim_Offset;

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

% Staircase Results -------------------------------

Abortion_Blocks = 0;

for i = 1:Block_Num

    if(Abortion_Blocks == 1)

        Contrast_Final = NaN;

    else

        for j = 1:Run_Num

            if(Run_Seq{j,3} == i)

                if(isempty(Run_Seq{j,8}) == false)

                    Contrast_Last = Run_Seq{j,8};

                    if(Run_Seq{j,2} == 4)

                        Abortion_Blocks = 1;

                    end

                else

                    break;

                end

            end

        end

        if(Abortion_Blocks == 1)

            Contrast_Final = Contrast_Last;

        else

            % Staircase Processing -------------------------------

            % Processing_Run_Seq : Target Oriention, Answer, Contrast

            Processing_Run_Seq = Run_Seq(j-Straircase_Step_Run_Num:j-1,[6 17 8]);

            Processing_Count = 0;

            for l = 1:Straircase_Step_Run_Num

                if(Processing_Run_Seq{l,1} == 45) && ...
                        (strcmp(Processing_Run_Seq{l,2}, 'LeftShift'))

                    Processing_Count = Processing_Count +1;

                elseif(Processing_Run_Seq{l,1} == -45) && ...
                        (strcmp(Processing_Run_Seq{l,2}, 'RightShift'))

                    Processing_Count = Processing_Count +1;

                end

            end

            if(Processing_Count >= (Straircase_Step_Run_Num * 0.8))

                Contrast_Final = 0.5 * Contrast_Last;

            elseif(Processing_Count < (Straircase_Step_Run_Num * 0.7))

                Contrast_Final = 2 * Contrast_Last;

                if(Contrast_Final> Initial_Contrast)

                    Contrast_Final = Initial_Contrast;

                end

            else

                Contrast_Final = Contrast_Last;

            end

            % ------------------------------------------------------

        end

    end

    Staircase_Results{i,3} = Contrast_Final;

end

Contrast_Threshold = mean(vertcat(Staircase_Results{:,3}), 1, "omitnan");

% ------------------------------------------------

%% saving and cleaning up

cfgOutput.Output_table = cell2table(Run_Seq,"VariableNames",["ID", "State", "Block_Number", ...
    "Block_Question_(Attention_Direction)", "Straircase_Step", "Target_Oriention", ...
    "Repetition", "Contrast", "ITI", "ISI", "Trial_Onset", "Cue_Onset", ...
    "Cue_Offset", "Stim_Onset", "Stim_Offset", "Response_Time", "Answer"]);

cfgOutput.Staircase_Results = Staircase_Results;

cfgOutput.Contrast_Threshold = Contrast_Threshold;

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
    % writetable(cfgOutput.Output_table,[cfgFile.subDir, cfgFile.BIDSname, cfgFile.csvFile]);
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