% Clear the workspace and the screen
sca;
close all;
clear;

Line_Lenghts = 20:23;
Shift_Directions = {'Right', 'Left'};
Block_Run_Num = 240;
Block_Repetition_Num = 2; % Number of Pair Shorter/Longer Blocks

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

Initial_Shift_Size = 1; % In Visual Degrees
Line_Thickness = angle2pix(cfgScreen,0.1);
Transecting_Line_Size = angle2pix(cfgScreen,0.2);
Transecting_Line_Thickness = angle2pix(cfgScreen,0.1);

Stim_Time = 0.2;
Response_Timeout = 2;

KbName('UnifyKeyNames');
Keyboard.quitKey = KbName('ESCAPE');
Keyboard.confirmKey = KbName('c');

Keyboard.Rightkey = KbName('RightArrow'); % Right
Keyboard.Leftkey = KbName('LeftArrow'); % Left
Keyboard.Neutralkey = KbName('DownArrow'); % Neutral

Noise_Contrast = 0.5;

Block_Num = 2 * Block_Repetition_Num;
Run_Num = Block_Run_Num * Block_Num;
number_of_short_breaks = 6;  % break every 160 trials ~ 10 min
number_of_big_breaks = 2;

Small_Break_Interval = Run_Num / number_of_short_breaks; % 2 Min
Big_Break_Interval = Run_Num / number_of_big_breaks; % 5 Min 

% ------------------------------------------------------------------------
% settings

prompt = {'Subject ID:', 'Session', 'Task', 'Run', 'Eyetracker? y/n', 'Skip sync test? y/n', 'First Block Question?'};
dlgtitle = 'Details';
defaults = {'','01','Landmark','01','n','n', 'Shorter'}; % you can put in default responses
opts.Interpreter = 'tex';
dims = [1, 40; 1, 40; 1, 40; 1, 40; 1, 40; 1, 40; 1, 40];
ansr = inputdlg(prompt, 'Info',dims,defaults,opts); % opens dialog
cfgExp.answer = cell2struct(ansr, {'sub','ses','task','run','eyetracker','skipSync', 'First_Block_Question'}, 1);

if strcmp(cfgExp.answer.eyetracker,'y'); cfgEyelink.on = 1; else, cfgEyelink.on = 0; end
if strcmp(cfgExp.answer.skipSync,'y'); skipSync = 1; else, skipSync = 0; end
if strcmp(cfgExp.answer.First_Block_Question,'Shorter'); First_Block_Question = 1; else, First_Block_Question = 0; end

if(skipSync)

    Screen('Preference', 'SkipSyncTests', 1);

else

    Screen('Preference', 'SkipSyncTests', 0);

end

cfgFile = create_file_directory(cfgExp);  % create file directories

Line_Lenghts = num2cell(Line_Lenghts);

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

        Run_Block_Questions{i,1} = "Shorter";

    else

        Run_Block_Questions{i,1} = "Longer";

    end

end

Run_Factors = cell(0);

for i = 1:Run_Num

    if (i == Run_Num)

        [Run_Line_Lenghts, Run_Shift_Directions] = ...
            BalanceTrials(Block_Run_Num, 1, Line_Lenghts, Shift_Directions);

        Block_Run_Factors = {Run_Line_Lenghts, Run_Shift_Directions};
        Block_Run_Factors = horzcat(Block_Run_Factors{:});

        Run_Factors = {Run_Factors, Block_Run_Factors};
        Run_Factors = vertcat(Run_Factors{:});

    elseif ((Run_Blocks{i,1} ~= Run_Blocks{i+1,1}))

        [Run_Line_Lenghts, Run_Shift_Directions] = ...
            BalanceTrials(Block_Run_Num, 1, Line_Lenghts, Shift_Directions);

        Block_Run_Factors = {Run_Line_Lenghts, Run_Shift_Directions};
        Block_Run_Factors = horzcat(Block_Run_Factors{:});

        Run_Factors = {Run_Factors, Block_Run_Factors};
        Run_Factors = vertcat(Run_Factors{:});

    end

end

ITIs = zeros(Run_Num,1);

for i = 1:Run_Num

    ITIs(i,1) = rand + 1; % Jitters ITI between 1 and 2 seconds

end

ITIs = num2cell(ITIs);

Run_Seq = {IDs, States, Run_Blocks, Run_Block_Questions, Run_Factors, ...
    num2cell(zeros(Run_Num,1)), ITIs, num2cell(zeros(Run_Num,4)), ...
    cellstr(strings(Run_Num,1))};

Run_Seq = horzcat(Run_Seq{:});

% Run_Seq : ID, State, Block Number, Block Question, Line Lenght,
% Shift Direction, Shift Size, ITI, Trial_Onset,
% Stim_Onset, Stim_Offset, Response Time, Answer

% State :
%
% 1: Done
% 2: -
% 3: No Answer
% 4: Abortion

[Staircase_Processing_Line_Lenghts, Staircase_Processing_Shift_Directions] = ...
    BalanceFactors(1, 0, Line_Lenghts, Shift_Directions);

Staircase_Processing = {Staircase_Processing_Line_Lenghts, ...
    Staircase_Processing_Shift_Directions, ...
    num2cell(zeros(size(Staircase_Processing_Line_Lenghts,1),1)), ...
    num2cell(repelem(Initial_Shift_Size,size(Staircase_Processing_Line_Lenghts,1))');};

Staircase_Processing = horzcat(Staircase_Processing{:});

% Staircase_Processing : Line Lenght, Shift Direction,
% Consecutive Corrects Count, Shift Size

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

% Noise ------------------------------------------

[Noise_Id, Noise_Rec] = ...
    CreateProceduralNoise(window, screenXpixels, screenYpixels, ...
    'Perlin', [0.5 0.5 0.5 0]);

% Fixation Cross ---------------------------------

% Size of the fixation cross
FixCross_Size_Pix = 34;

FixCross_xCoords = [-FixCross_Size_Pix/2 FixCross_Size_Pix/2 0 0];
FixCross_yCoords = [0 0 -FixCross_Size_Pix/2 FixCross_Size_Pix/2];
FixCross_Coords = [FixCross_xCoords; FixCross_yCoords];

% Set the line width for the fixation cross
FixCross_Thickness_Pix = 4;

% Transecting Line ---------------------------------

Transecting_Line_xCoords = [0 0];
Transecting_Line_yCoords = [-Transecting_Line_Size/2 Transecting_Line_Size/2];
Transecting_Line_Coords = [Transecting_Line_xCoords; Transecting_Line_yCoords];

% -------------------------------------------------

% Measure the vertical refresh rate of the monitor
ifi = Screen('GetFlipInterval', window);

Stim_Frames = round(Stim_Time / ifi);
Response_Timeout_Frames = round(Response_Timeout / ifi);

% Setup the text type for the window
Screen('TextFont', window, 'Ariel');
Screen('TextSize', window, 32);

DrawFormattedText(window, 'Press Anykey To Start :)', 'center', 'center',[1 1 1]);

% Flip to the screen
Screen('Flip', window);

% Wait for a key press
KbStrokeWait;

ListenChar(-1); % makes it so characters typed dont show up in the command window
HideCursor(); % hides the cursor

Keyboard.activeKeys = [Keyboard.quitKey, Keyboard.confirmKey, Keyboard.Rightkey, Keyboard.Leftkey, Keyboard.Neutralkey];
Keyboard.responseKeys = [Keyboard.Rightkey, Keyboard.Leftkey, Keyboard.Neutralkey];
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

        DrawFormattedText(window, 'Take a break For 5 Min :)', 'center', 'center',[1 1 1]);
        Screen('Flip',window); % swaps backbuffer to frontbuffer

        DrawFormattedText(window, 'Press Anykey To Start :)', 'center', 'center',[1 1 1]);

        WaitSecs(300);

        Screen('Flip',window);

        % Wait for a key press
        KbStrokeWait;

    elseif ((ceil(n / Small_Break_Interval) ~= ceil((n-1) / Small_Break_Interval)) && n ~= 1)

        DrawFormattedText(window, 'Take a break For 1 Min :)', 'center', 'center',[1 1 1]);
        Screen('Flip',window); % swaps backbuffer to frontbuffer

        DrawFormattedText(window, 'Press Anykey To Start :)', 'center', 'center',[1 1 1]);

        WaitSecs(60);

        Screen('Flip',window);

        % Wait for a key press
        KbStrokeWait;

    end

    % Block Check -----------------------------------------------------

    if (n == 1)

        if (strcmp(Run_Seq{n,4}, 'Shorter'))

            DrawFormattedText(window, 'Next Block: Which Side Is Shorter?', 'center', 'center',[1 1 1]);
            send_trigger(cfgEyelink, 'Shorter Block');

        elseif (strcmp(Run_Seq{n,4}, 'Longer'))

            DrawFormattedText(window, 'Next Block: Which Side Is Longer?', 'center', 'center',[1 1 1]);
            send_trigger(cfgEyelink, 'Longer Block');

        end

        Screen('Flip',window); % swaps backbuffer to frontbuffer

        DrawFormattedText(window, 'Press Anykey To Start :)', 'center', 'center',[1 1 1]);

        WaitSecs(4);

        Screen('Flip',window);

        % Wait for a key press
        KbStrokeWait;

    elseif (Run_Seq{n,3} ~= Run_Seq{n-1,3})

        Staircase_Processing = {Staircase_Processing_Line_Lenghts, ...
            Staircase_Processing_Shift_Directions, ...
            num2cell(zeros(size(Staircase_Processing_Line_Lenghts,1),1)), ...
            num2cell(repelem(Initial_Shift_Size,size(Staircase_Processing_Line_Lenghts,1))');};

        Staircase_Processing = horzcat(Staircase_Processing{:});

        if (strcmp(Run_Seq{n,4}, 'Shorter'))

            DrawFormattedText(window, 'Next Block: Which Side Is Shorter?', 'center', 'center',[1 1 1]);
            send_trigger(cfgEyelink, 'Shorter Block');


        elseif (strcmp(Run_Seq{n,4}, 'Longer'))

            DrawFormattedText(window, 'Next Block: Which Side Is Longer?', 'center', 'center',[1 1 1]);
            send_trigger(cfgEyelink, 'Longer Block');

        end

        Screen('Flip',window); % swaps backbuffer to frontbuffer

        DrawFormattedText(window, 'Press Anykey To Start :)', 'center', 'center',[1 1 1]);

        WaitSecs(4);

        Screen('Flip',window);

        % Wait for a key press
        KbStrokeWait;

    end

    % Staircase Processing -------------------------------

    for i = 1:size(Staircase_Processing,1)

        if ((Staircase_Processing{i,1} == Run_Seq{n,5}) && ...
                (strcmp(Staircase_Processing{i,2}, Run_Seq{n,6})))

            Run_Seq{n,7} = Staircase_Processing{i,4};

        end

    end

    % Line -----------------------------------------------

    Line_Size = angle2pix(cfgScreen,Run_Seq{n,5});

    Line_xCoords = [-Line_Size/2 Line_Size/2];
    Line_yCoords = [0 0];
    Line_Coords = [Line_xCoords; Line_yCoords];

    if (strcmp(Run_Seq{n,6}, 'Right'))

        % Shift to Right: Right is Longer
        Line_Center = [xCenter + angle2pix(cfgScreen,Run_Seq{n,7}), yCenter];

    elseif (strcmp(Run_Seq{n,6}, 'Left'))

        Line_Center = [xCenter - angle2pix(cfgScreen,Run_Seq{n,7}), yCenter];

    end

    % ----------------------------------------------------

    % ITI

    ITI_Frames = round(Run_Seq{n,8} / ifi);

    for frame = 1:ITI_Frames

        % Draw the fixation cross
        Screen('DrawLines', window, FixCross_Coords, FixCross_Thickness_Pix, [1 1 1], [xCenter yCenter], 2);

        if (frame == 1)
            Trial_Onset = Screen('Flip',window); % swaps backbuffer to frontbuffer
            vbl = Trial_Onset;
        else
            vbl = Screen('Flip', window, vbl + (0.5 * ifi));
        end

    end

    % Stim

    for frame = 1:Stim_Frames

        % Draw the Transecting Line
        Screen('DrawLines', window, Transecting_Line_Coords, Transecting_Line_Thickness, [1 1 1], [xCenter yCenter], 2);

        % Draw the Line
        Screen('DrawLines', window, Line_Coords, Line_Thickness, [1 1 1], Line_Center, 2);

        if (frame == 1)

            Stim_Onset = Screen('Flip',window); % swaps backbuffer to frontbuffer
            vbl = Stim_Onset;
            KbQueueFlush; % Flushes Buffer

            if (strcmp(Run_Seq{n,6}, 'Right'))

                send_trigger(cfgEyelink, 'Right Shift');

            elseif (strcmp(Run_Seq{n,6}, 'Left'))

                send_trigger(cfgEyelink, 'Left Shift');

            end

        else
            vbl = Screen('Flip', window, vbl + (0.5 * ifi));
        end

    end

    % Set new noise seed value
    seed = randi([1000,1000000]);

    % Disable alpha-blending for Noise
    Screen('BlendFunction', window, 'GL_ONE', 'GL_ZERO');

    % Noise
    Screen('DrawTexture', window, Noise_Id, [], [], [], [], [], [], ...
        [], [], [Noise_Contrast, seed, 0, 0]);

    % Set up alpha-blending (Global)
    Screen('BlendFunction', window, 'GL_SRC_ALPHA', 'GL_ONE_MINUS_SRC_ALPHA');

    Stim_Offset = Screen('Flip',window);

    noResp = 1;

    while (noResp == 1)

        [presd, firstPrsd] = KbQueueCheck;  % listens for response
        keyCod = find(firstPrsd,1);  % collects the pressed key code

        if (presd && (ismember(keyCod,Keyboard.responseKeys))) % store response variables

            send_trigger(cfgEyelink, 'Response');
            Response_Key_Time = firstPrsd(keyCod);  % exact time of button press
            Key = KbName(keyCod);  % which key was pressed
            Key = string(Key);

%             old = 'RightArrow';
%             new = 'Right';
%             Key = replace(Key,old,new);
%             old = 'LeftArrow';
%             new = 'Left';
%             Key = replace(Key,old,new);
%             old = 'DownArrow';
%             new = 'Neutral';
%             Key = replace(Key,old,new);

            if (strcmp(Key, 'RightArrow'))

                send_trigger(cfgEyelink, 'Right Response');

            elseif (strcmp(Key, 'LeftArrow'))

                send_trigger(cfgEyelink, 'Left Response');

            elseif (strcmp(Key, 'DownArrow'))

                send_trigger(cfgEyelink, 'Neutral Response');

            end

            Run_Seq{n,12} = Response_Key_Time;
            Run_Seq{n,13} = Key ;
            Run_Seq{n,2} = 1; % 1: Done

            noResp = 0;
            break;

        elseif (presd && keyCod == Keyboard.quitKey)

            warning('Experiment Aborted!')
            Abortion_Pauses(n,1) = Abortion_Pauses(n,1) + 1;
            send_trigger(cfgEyelink, 'Pause');

            DrawFormattedText(window, 'Press C to confirm exit or any other key to continue', 'center', 'center',[1 1 1]);
            Screen('Flip',window);

            [~, abrtPrsd] = KbStrokeWait;
            if abrtPrsd(Keyboard.confirmKey)

                Abortion = 1;
                Run_Seq{n,2} = 4; % 4: Abortion
                Run_Seq{n,12} = NaN;
                Run_Seq{n,13} = 'None' ;

                noResp = 0;
                break;

            end

            % -----------------------------------------
            % Repeating Trial

            send_trigger(cfgEyelink, 'Repeating Trial');

            % ITI

            for frame = 1:ITI_Frames

                % Draw the fixation cross
                Screen('DrawLines', window, FixCross_Coords, FixCross_Thickness_Pix, [1 1 1], [xCenter yCenter], 2);

                if (frame == 1)
                    Trial_Onset = Screen('Flip',window); % swaps backbuffer to frontbuffer
                    vbl = Trial_Onset;
                else
                    vbl = Screen('Flip', window, vbl + (0.5 * ifi));
                end

            end

            % Stim

            for frame = 1:Stim_Frames

                % Draw the Transecting Line
                Screen('DrawLines', window, Transecting_Line_Coords, Transecting_Line_Thickness, [1 1 1], [xCenter yCenter], 2);

                % Draw the Line
                Screen('DrawLines', window, Line_Coords, Line_Thickness, [1 1 1], Line_Center, 2);

                if (frame == 1)

                    Stim_Onset = Screen('Flip',window); % swaps backbuffer to frontbuffer
                    vbl = Stim_Onset;
                    KbQueueFlush; % Flushes Buffer

                    if (strcmp(Run_Seq{n,6}, 'Right'))

                        send_trigger(cfgEyelink, 'Right Shift');

                    elseif (strcmp(Run_Seq{n,6}, 'Left'))

                        send_trigger(cfgEyelink, 'Left Shift');

                    end

                else
                    vbl = Screen('Flip', window, vbl + (0.5 * ifi));
                end

            end

            % Set new noise seed value
            seed = randi([1000,1000000]);

            % Disable alpha-blending for Noise
            Screen('BlendFunction', window, 'GL_ONE', 'GL_ZERO');

            % Noise
            Screen('DrawTexture', window, Noise_Id, [], [], [], [], [], [], ...
                [], [], [Noise_Contrast, seed, 0, 0]);

            % Set up alpha-blending (Global)
            Screen('BlendFunction', window, 'GL_SRC_ALPHA', 'GL_ONE_MINUS_SRC_ALPHA');

            Stim_Offset = Screen('Flip',window);

            % -----------------------------------------

        elseif ((GetSecs - Stim_Offset) > Response_Timeout)  % Stop listening

            Run_Seq{n,2} = 3; % 3: No Answer
            Run_Seq{n,12} = NaN;
            Run_Seq{n,13} = 'None' ;

            noResp = 0;
            break;

        end

    end

    Run_Seq{n,9} = Trial_Onset;
    Run_Seq{n,10} = Stim_Onset;
    Run_Seq{n,11} = Stim_Offset;

    % Staircase Processing -------------------------------

    for i = 1:size(Staircase_Processing,1)

        if ((Staircase_Processing{i,1} == Run_Seq{n,5}) && ...
                (strcmp(Staircase_Processing{i,2}, Run_Seq{n,6})))

            if(strcmp(Run_Seq{n,4}, 'Longer'))

                if(strcmp(Run_Seq{n,6}, Run_Seq{n,13}))

                    Staircase_Processing{i,3} = Staircase_Processing{i,3} +1;

                else

                    Staircase_Processing{i,3} = 0;

                end

            elseif(strcmp(Run_Seq{n,4}, 'Shorter'))

                if(((strcmp(Run_Seq{n,6}, 'Right')) && (strcmp(Run_Seq{n,13}, 'Left'))) || ...
                        ((strcmp(Run_Seq{n,6}, 'Left')) && (strcmp(Run_Seq{n,13}, 'Right'))))

                    Staircase_Processing{i,3} = Staircase_Processing{i,3} +1;

                else

                    Staircase_Processing{i,3} = 0;

                end

            end

            if (Staircase_Processing{i,3} == 2)

                Staircase_Processing{i,4} = 0.8 * Staircase_Processing{i,4};
                Staircase_Processing{i,3} = 0;

            elseif(Staircase_Processing{i,3} == 0)

                Staircase_Processing{i,4} = 1.25 * Staircase_Processing{i,4};

                if(Staircase_Processing{i,4}> Initial_Shift_Size)

                    Staircase_Processing{i,4} = Initial_Shift_Size;

                end

            end

        end

    end

    % ----------------------------------------------------

end

Task_Offset = send_trigger(cfgEyelink, 'End of Experiment');

DrawFormattedText(window, 'Press Anykey To Exit :)', 'center', 'center',[1 1 1]);

Screen('Flip',window); % swaps backbuffer to frontbuffer

% Wait for a key press
KbStrokeWait;

ListenChar(0); % Makes it so characters typed do show up in the command window
ShowCursor(); % Shows the cursor
Screen('CloseAll'); % Closes Screen

% Clear the screen
sca;

%% saving and cleaning up

cfgOutput.Output_table = cell2table(Run_Seq,"VariableNames",["ID", "State", ...
    "Block_Number", "Block_Question", "Line_Lenght", "Shift_Direction", ...
    "Shift_Size", "ITI", "Trial_Onset", "Stim_Onset", ...
    "Stim_Offset", "Response Time", "Answer"]);

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