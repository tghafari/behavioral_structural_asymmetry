% Clear the workspace and the screen
sca;
close all;
clear;

Line_Lenghts = 20:1:23;
Shift_Directions = {'Right', 'Left'};
Straircase_Step_Num = 10;
Block_Repetition_Num = 2; % Number of Pair Shorter/Longer Blocks

% Screen properties
PsychDefaultSetup(2);
cfgScreen.scrNum = max(Screen('Screens'));  % get screen number - draw to the external screen if avaliable

[cfgScreen.dispSize.width, cfgScreen.dispSize.height]...
    = Screen('DisplaySize', cfgScreen.scrNum);  % get the physical size of the screen in millimeters
cfgScreen.distance = 60;  % set the distance from participant to the monitor in cm
cfgScreen.resolution = Screen('Resolution', cfgScreen.scrNum);  % get/set the on screen resolution

Initial_Shift_Size = 1; % In Visual Degrees
Line_Thickness = angle2pix(cfgScreen,0.1);
Transecting_Line_Size = angle2pix(cfgScreen,0.2);
Transecting_Line_Thickness = angle2pix(cfgScreen,0.1);

Stim_Time = 0.2;
Response_Timeout = 2;

KbName('UnifyKeyNames');
Keyboard.quitKey = KbName('ESCAPE');
Keyboard.confirmKey = KbName('c');

Keyboard.Rightkey = KbName('RightShift'); % Right
Keyboard.Leftkey = KbName('LeftShift'); % Left
Keyboard.Neutralkey = KbName('space'); % Neutral

Noise_Contrast = 0.5;

Block_Num = 2 * Block_Repetition_Num;
Line_Lenght_Num = size(Line_Lenghts,2);
Shift_Direction_Num = size(Shift_Directions,2);

Straircase_Step_Run_Num = Line_Lenght_Num * Shift_Direction_Num;
Block_Run_Num = Straircase_Step_Run_Num * Straircase_Step_Num;
Run_Num = Block_Run_Num * Block_Num;

Small_Break_Interval = Run_Num / Block_Num; % 1 Min
Big_Break_Interval = Run_Num + 1; % 10 Min (Inactive)

% ------------------------------------------------------------------------

prompt = {'Enter ID:'}; % description of fields
defaults = {'',''}; % you can put in default responses
opts.Interpreter = 'tex';
dims = [1 40];
answer = inputdlg(prompt, 'Info',dims,defaults,opts); % opens dialog
subject = answer{1,:};
clock_info = clock; % Current date and time as date vector. [year month day hour minute seconds]
Output_Name=[subject '_' num2str(clock_info(2)) '_' num2str(clock_info(3)) '_' num2str(clock_info(4)) '_' num2str(clock_info(5))]; % makes unique filename

Line_Lenghts = num2cell(Line_Lenghts);

IDs = 1:1:Run_Num;
IDs = num2cell(IDs');
States = zeros(Run_Num,1);
States = num2cell(States);

Blocks = 1:1:Block_Num;
Run_Blocks = repelem(Blocks,Block_Run_Num);
Run_Blocks = num2cell(Run_Blocks');

Run_Block_Questions = cell(Run_Num,1);

for i = 1:Run_Num

    if (mod(Run_Blocks{i,1},2) == 1)

        Run_Block_Questions{i,1} = "Shorter";

    elseif (mod(Run_Blocks{i,1},2) == 0)

        Run_Block_Questions{i,1} = "Longer";

    end

end

Straircase_Steps = 1:1:Straircase_Step_Num;
Block_Straircase_Steps = repelem(Straircase_Steps,Straircase_Step_Run_Num);
Run_Straircase_Steps = repmat(Block_Straircase_Steps,1,Block_Num);
Run_Straircase_Steps = num2cell(Run_Straircase_Steps');

Run_Factors = cell(0);

for i = 1:Run_Num

    if (i == Run_Num)

        [Run_Line_Lenghts, Run_Shift_Directions] = ...
            BalanceFactors(1, 1, Line_Lenghts, Shift_Directions);

        Run_Shift_Sizes = cell(Straircase_Step_Run_Num,1);

        if (Run_Straircase_Steps{i,1} == 1)

            Run_Shift_Sizes = num2cell(repelem(Initial_Shift_Size,Straircase_Step_Run_Num)');

        end

        Straircase_Step_Run_Factors = {Run_Line_Lenghts, Run_Shift_Directions, Run_Shift_Sizes};
        Straircase_Step_Run_Factors = horzcat(Straircase_Step_Run_Factors{:});

        Run_Factors = {Run_Factors, Straircase_Step_Run_Factors};
        Run_Factors = vertcat(Run_Factors{:});

    elseif ((Run_Straircase_Steps{i,1} ~= Run_Straircase_Steps{i+1,1}))

        [Run_Line_Lenghts, Run_Shift_Directions] = ...
            BalanceFactors(1, 1, Line_Lenghts, Shift_Directions);

        Run_Shift_Sizes = cell(Straircase_Step_Run_Num,1);

        if (Run_Straircase_Steps{i,1} == 1)

            Run_Shift_Sizes = num2cell(repelem(Initial_Shift_Size,Straircase_Step_Run_Num)');

        end

        Straircase_Step_Run_Factors = {Run_Line_Lenghts, Run_Shift_Directions, Run_Shift_Sizes};
        Straircase_Step_Run_Factors = horzcat(Straircase_Step_Run_Factors{:});

        Run_Factors = {Run_Factors, Straircase_Step_Run_Factors};
        Run_Factors = vertcat(Run_Factors{:});

    end

end

ITIs = zeros(Run_Num,1);

for i = 1:Run_Num

    ITIs(i,1) = rand + 1; % Jitters ITI between 1 and 2 seconds

end

ITIs = num2cell(ITIs);

Run_Seq = {IDs, States, Run_Blocks, Run_Block_Questions, ...
    Run_Straircase_Steps, Run_Factors, ITIs, ...
    num2cell(zeros(Run_Num,4)),cellstr(strings(Run_Num,1))};

Run_Seq = horzcat(Run_Seq{:});

% Run_Seq : ID, State, Block Number, Block Question, Straircase Step,
% Line Lenght, Shift Direction, Shift Size, ITI, Trial_Onset,
% Stim_Onset, Stim_Offset, RT, Answer

% State :
%
% 1: Done
% 2: -
% 3: No Answer
% 4: Abortion

% ------------------------------------------------------------------------

PsychDefaultSetup(2);

% Get the screen numbers
screens = Screen('Screens');

% Draw to the external screen if avaliable
screenNumber = max(screens);

white = WhiteIndex(screenNumber);
black = BlackIndex(screenNumber);
grey = (white - black) / 2;

% Open an on screen window
[window, windowRect] = PsychImaging('OpenWindow', screenNumber, grey);

% Maximum priority
topPriorityLevel = MaxPriority(window);
Priority(topPriorityLevel);

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

for n = 1:Run_Num

    if (Abortion == 1)

        break;

    end

    % Break Check -----------------------------------------------------

    if ((ceil(n / Big_Break_Interval) ~= ceil((n-1) / Big_Break_Interval)) && n ~= 1)

        DrawFormattedText(window, 'Take a break For 10 Min :)', 'center', 'center',[1 1 1]);
        vbl=Screen('Flip',window); % swaps backbuffer to frontbuffer

        DrawFormattedText(window, 'Press Anykey To Start :)', 'center', 'center',[1 1 1]);

        Screen('Flip',window,vbl + 600);

        % Wait for a key press
        KbStrokeWait;

    elseif ((ceil(n / Small_Break_Interval) ~= ceil((n-1) / Small_Break_Interval)) && n ~= 1)

        DrawFormattedText(window, 'Take a break For 1 Min :)', 'center', 'center',[1 1 1]);
        vbl=Screen('Flip',window); % swaps backbuffer to frontbuffer

        DrawFormattedText(window, 'Press Anykey To Start :)', 'center', 'center',[1 1 1]);

        Screen('Flip',window,vbl + 60);

        % Wait for a key press
        KbStrokeWait;

    end

    % Block Check -----------------------------------------------------

    if (n == 1)

        if (strcmp(Run_Seq{n,4}, 'Shorter'))

            DrawFormattedText(window, 'Next Block: Which Side Is Shorter?', 'center', 'center',[1 1 1]);

        elseif (strcmp(Run_Seq{n,4}, 'Longer'))

            DrawFormattedText(window, 'Next Block: Which Side Is Longer?', 'center', 'center',[1 1 1]);

        end

        vbl=Screen('Flip',window); % swaps backbuffer to frontbuffer

        DrawFormattedText(window, 'Press Anykey To Start :)', 'center', 'center',[1 1 1]);

        Screen('Flip',window,vbl + 4);

        % Wait for a key press
        KbStrokeWait;

    elseif (Run_Seq{n,3} ~= Run_Seq{n-1,3})

        if (strcmp(Run_Seq{n,4}, 'Shorter'))

            DrawFormattedText(window, 'Next Block: Which Side Is Shorter?', 'center', 'center',[1 1 1]);

        elseif (strcmp(Run_Seq{n,4}, 'Longer'))

            DrawFormattedText(window, 'Next Block: Which Side Is Longer?', 'center', 'center',[1 1 1]);

        end

        vbl=Screen('Flip',window); % swaps backbuffer to frontbuffer

        DrawFormattedText(window, 'Press Anykey To Start :)', 'center', 'center',[1 1 1]);

        Screen('Flip',window,vbl + 4);

        % Wait for a key press
        KbStrokeWait;

    end

    % Staircase Processing -------------------------------

    if (isempty(Run_Seq{n,8}))

        % Processing_Run_Seq : Line Lenght, Block Question,
        % Shift Direction, Answer, Shift Size

        Processing_Run_Seq = Run_Seq(n-Straircase_Step_Run_Num:n-1,[6 4 7 14 8]);

        for i = 1:Line_Lenght_Num

            Processing_Count = 0;

            for j = 1:Straircase_Step_Run_Num

                if(Processing_Run_Seq{j,1} == Line_Lenghts{1,i})

                    if((strcmp(Processing_Run_Seq{j,2}, 'Longer')) && ...
                            (strcmp(Processing_Run_Seq{j,3}, Processing_Run_Seq{j,4})))

                        Processing_Count = Processing_Count +1;

                    elseif(strcmp(Processing_Run_Seq{j,2}, 'Shorter'))

                        if((strcmp(Processing_Run_Seq{j,3}, 'Right')) && ...
                                (strcmp(Processing_Run_Seq{j,4}, 'Left')))

                            Processing_Count = Processing_Count +1;

                        end

                        if((strcmp(Processing_Run_Seq{j,3}, 'Left')) && ...
                                (strcmp(Processing_Run_Seq{j,4}, 'Right')))

                            Processing_Count = Processing_Count +1;

                        end

                    end

                    Current_Shift = Processing_Run_Seq{j,5};

                end

            end

            if (Processing_Count == 2)

                for k = n:(n+Straircase_Step_Run_Num-1)

                    if(Run_Seq{k,6} == Line_Lenghts{1,i})

                        Run_Seq{k,8} = 0.8 * Current_Shift;

                    end

                end

            elseif(Processing_Count < 2)

                for k = n:(n+Straircase_Step_Run_Num-1)

                    if(Run_Seq{k,6} == Line_Lenghts{1,i})

                        Run_Seq{k,8} = 1.25 * Current_Shift;

                        if(Run_Seq{k,8}> Initial_Shift_Size)

                            Run_Seq{k,8} = Initial_Shift_Size;

                        end

                    end

                end

            end

        end

    end

    % Line -----------------------------------------------

    Line_Size = angle2pix(cfgScreen,Run_Seq{n,6});

    Line_xCoords = [-Line_Size/2 Line_Size/2];
    Line_yCoords = [0 0];
    Line_Coords = [Line_xCoords; Line_yCoords];

    if (strcmp(Run_Seq{n,7}, 'Right'))

        Line_Center = [xCenter + angle2pix(cfgScreen,Run_Seq{n,8}), yCenter];

    elseif (strcmp(Run_Seq{n,7}, 'Left'))

        Line_Center = [xCenter - angle2pix(cfgScreen,Run_Seq{n,8}), yCenter];

    end

    % ----------------------------------------------------

    ITI_Frames = round(Run_Seq{n,9} / ifi);

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
        else
            vbl = Screen('Flip', window, vbl + (0.5 * ifi));
        end

    end

    % Set new noise seed value
    seed = randi([1000,10000]);

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

            Response_Key_Time = firstPrsd(keyCod);  % exact time of button press
            Key = KbName(keyCod);  % which key was pressed
            Key = string(Key);

            old = '4$';
            new = 'Right';
            Key = replace(Key,old,new);
            old = '6^';
            new = 'Left';
            Key = replace(Key,old,new);
            old = '5%';
            new = 'Neutral';
            Key = replace(Key,old,new);

            Run_Seq{n,13} = Response_Key_Time-Stim_Onset ;
            Run_Seq{n,14} = Key ;
            Run_Seq{n,2} = 1; % 1: Done

            noResp = 0;
            break;

        elseif (presd && keyCod == Keyboard.quitKey)

            warning('Experiment Aborted!')
            Abortion_Pauses(n,1) = Abortion_Pauses(n,1) + 1;

            DrawFormattedText(window, 'Press C To Confirm :)', 'center', 'center',[1 1 1]);
            Screen('Flip',window);

            [~, abrtPrsd] = KbStrokeWait;
            if abrtPrsd(Keyboard.confirmKey)

                Abortion = 1;
                Run_Seq{n,2} = 4; % 4: Abortion
                Run_Seq{n,13} = NaN;
                Run_Seq{n,14} = 'None' ;

                noResp = 0;
                break;

            end

            % -----------------------------------------
            % Repeating Trial

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
                else
                    vbl = Screen('Flip', window, vbl + (0.5 * ifi));
                end

            end

            % Set new noise seed value
            seed = randi([1000,10000]);

            % Disable alpha-blending for Noise
            Screen('BlendFunction', window, 'GL_ONE', 'GL_ZERO');

            % Noise
            Screen('DrawTexture', window, Noise_Id, [], [], [], [], [], [], [], [], [Noise_Contrast, seed, 0, 0]);

            % Set up alpha-blending (Global)
            Screen('BlendFunction', window, 'GL_SRC_ALPHA', 'GL_ONE_MINUS_SRC_ALPHA');

            Stim_Offset = Screen('Flip',window);

            % -----------------------------------------

        elseif ((GetSecs - Stim_Offset) > Response_Timeout)  % Stop listening

            Run_Seq{n,2} = 3; % 3: No Answer
            Run_Seq{n,13} = NaN;
            Run_Seq{n,14} = 'None' ;

            noResp = 0;
            break;

        end

    end

    Run_Seq{n,10} = Trial_Onset;
    Run_Seq{n,11} = Stim_Onset;
    Run_Seq{n,12} = Stim_Offset;

end

DrawFormattedText(window, 'Press Anykey To Exit :)', 'center', 'center',[1 1 1]);

Screen('Flip',window); % swaps backbuffer to frontbuffer

% Wait for a key press
KbStrokeWait;

ListenChar(0); % Makes it so characters typed do show up in the command window
ShowCursor(); % Shows the cursor
Screen('CloseAll'); % Closes Screen

% Clear the screen
sca;

% ------------------------------------------------------------------------

Output_table = cell2table(Run_Seq,"VariableNames",["ID", "State", ...
    "Block Number", "Block Question", "Straircase Step", "Line Lenght", ...
    "Shift Direction", "Shift Size", "ITI", "Trial_Onset", "Stim_Onset", ...
    "Stim_Offset", "RT", "Answer"]);

writetable(Output_table,strcat('Data/Subject_', Output_Name,'.csv'));
save(strcat('Data/Subject_', Output_Name));

Priority(0);