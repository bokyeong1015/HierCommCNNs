clc; clear; close all

%% General Setting
curPath = pwd;

%%%%%%%%%%%%%%%% Setting for data loading
% dataFolder = 'dataFull_Final_SFEW_2_Test'; % folder name containing raw images
dataFolder = 'dataSample_Final_SFEW_2_Test'; % folder name containing raw images
imgFormat_str = 'png'; % image format, e.g., png, gif, jpeg
dataType_str = 'test'; % data type e.g., train, valid, test

%%%%%%%%%%%%%%%% Setting for publicly available models for face detection
%%%%%%%%%%%%%%%% and landmark detection
module_ZRmodel_path = 'pipeline_modules_functions/module1_ZR_FaceDetector';
module_VJmodel_path = 'pipeline_modules_functions/module2_VJ_FaceDetector';
module_IFmodel_path = 'pipeline_modules_functions/module3_INTRAFACE_LandmarkDetector';
addpath('pipeline_modules_functions');

% if you individually use each model for another purpose, 
% please cite the corresponding reference
% [ZRmodel] Zhu, X., & Ramanan, D. 2012. Face detection, pose estimation, and landmark localization in the wild. In CVPR 2012, 2879-2886.
% : available @ http://www.ics.uci.edu/~xzhu/face/
% [VJmodel] Viola, P., & Jones, M. J. 2004. Robust real-time face detection. Int. J. Comput. Vision. 57(2), 137-154
% : available @ openCV haarcascades modules
% [IFmodel] Xiong, X., & De la Torre, F. 2013. Supervised descent method and its applications to face alignment. In CVPR 2013, 532-539
% : available @ http://www.humansensing.cs.cmu.edu/intraface/index.php

%%%%%%%%%%%%%%%% Setting for saving final aligned results
saveFolder = ['result_FullyAutoFaceAlign_',dataType_str]; % folder name for saving final alignments
if ~exist(saveFolder, 'dir'); mkdir(saveFolder); end

resize_w_h = 48; % size of final aligned face

%%%%%%%%%%%%%%%% Flag to Determine Showing Figures: 1 for true, 0 for false
%%%% intermediate figures from sub_Run1_1_4PipelineBased_FaceAlignment.m
fig_flag_faceDetect_1 = 0; % pipeline 1. Color IMG - VJ Model
fig_flag_faceDetect_2 = 0; % pipeline 2. Gray/HistEq IMG - VJ Model
fig_flag_lmkDetect_1 = 0; % pipeline 1. Color IMG - VJ Model - Gray IMG - IF Model
fig_flag_lmkDetect_2 = 0; % pipeline 2. Gray/HistEq IMG - VJ Model - Gray/HistEq IMG - IF Model
fig_flag_lmkDetect_3 = 0; % pipeline 3. Color IMG - ZR Model - Gray IMG - IF Model
fig_flag_lmkDetect_4 = 0; % pipeline 4. Color IMG - ZR Model - Gray/HistEq IMG - IF Model

%%%% final collective figures
fig_flag_all4pipelines = 1;
fig_flag_finalAlign = 1;

position_fig1 = [10 160 1000 800]; % figure position, all4pipelines, [left bottom width height]
position_fig2 = [10 200 700 200]; % figure position, finalAlign

%% Loading Data: Test Data

dataName_list = dir([dataFolder,'/*.',imgFormat_str]);
numSamples = size(dataName_list,1);    
origImg_collec = cell(numSamples, 2);

%% Setting for Face & Landmark Detection Models

%%%%%%%%%%%%%%%% ZR-Model Setup
addpath(module_ZRmodel_path);
load face_p146_small.mat

model.interval = 5; % 5 levels for each octave
model.thresh = min(-0.65, model.thresh); % set up the threshold

% define the mapping from view-specific mixture id to viewpoint
if length(model.components)==13; posemap = 90:-15:-90;
elseif length(model.components)==18; posemap = [90:-15:15 0 0 0 0 0 0 -15:-15:-90];
else error('Can not recognize this model');
end

%%%%%%%%%%%%%%%% VJ-Model Setup
addpath(module_VJmodel_path);

%%%%%%%%%%%%%%%% IntraFace Setup
addpath(module_IFmodel_path);
cd(module_IFmodel_path);

[Models, model_option] = xx_initialize; % IntraFace model setup
cd(curPath);


%% Collecting all raw images

disp('===============================================================')
disp(['Load Data: ', dataType_str, ' - total ',num2str(numSamples), ' samples'])

time_dataLoad_0 = tic;
for img_idx = 1:1:size(dataName_list,1)    
    
    temp_name = dataName_list(img_idx,1).name;
    
    if strcmp(temp_name, '.') ~= 1 && strcmp(temp_name, '..') ~= 1            
        raw_im = imread([dataFolder,'\',temp_name]);    
        origImg_collec{img_idx,1} = raw_im;
        origImg_collec{img_idx,2} = temp_name;
    end
    
    clear raw_im temp_name    
end
time_dataLoad = toc(time_dataLoad_0);
disp(['  loading time = ',num2str(time_dataLoad,3),' sec'])
disp(' ')

%% Multi-Pipeline Face Registration - Face & Landmark Detection

time_process_0 = tic;

for sample_idx = 1:1:size(origImg_collec,1)
    disp('===============================================================')
    disp('===============================================================')
    
    raw_img_color = origImg_collec{sample_idx, 1};    
    raw_img_gray = rgb2gray(raw_img_color);
    raw_img_histeq = histeq(raw_img_gray);
    
    orig_title = origImg_collec{sample_idx, 2};   
    
    %% 1/3: All 4 pipelines    
    disp(['All 4 pipelines: ',dataType_str,' s', num2str(sample_idx),'/',num2str(size(origImg_collec,1)),...,
        ' | ',orig_title])
    disp(' ')
    
    sub_Run1_1_4PipelineBased_FaceAlignment

    total_time_eachSample = time_fd_1 + time_fd_2 + time_fd_3 + time_fd_4 + ...
                            time_lmk_1 + time_lmk_2 + time_lmk_3 + time_lmk_4;
    disp(' ')
    disp(['Time for All 4 Pipelines: ',num2str(total_time_eachSample,3),' sec']);
    disp(' ')
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%      
    if fig_flag_all4pipelines == 1
        func_fig1_4PipelineBased_FaceAlignment
        
        saveas(gcf,[saveFolder,'\',...,
                    'fig_',dataType_str,'_s',num2str(sample_idx),'_',...,
                    orig_title(1:end-4),'_1.jpg'],'jpg');
    end   
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %% 2/3: Selection
    disp('---------------------------------------------------------------')
    disp(['Selection: ',dataType_str,' s', num2str(sample_idx),'/',num2str(size(origImg_collec,1))])
    disp(' ')        
    
    % Case 1. at least one pipeline successes in both Fd & LMKd
    % Case 2. at least one pipeline successes in Fd
    %  -- Case 2a. only one pipeline sucesses in Fd | must be 1 or 2
    %  -- Case 2b. more than 2 pipelines success in Fd
    %              must be a pair (1,2), (3,4) or (1,3,4), (2,3,4) or (1,2,3,4)
    %  ---- Case 2b-2. Fd via 2 VJ models
    %  ------ Case 2b-1b. Fd via 2 ZR models: Fail in Retrying        
    %  ------ Case 2b-1a. Fd via 2 ZR models: Success in Retrying
    % Case 3. all pipelines fail in Fd

    sub_Run1_2_finalSelect_MaxConfidenceResult
    
    disp(' ')
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%      
    if fig_flag_finalAlign == 1
        
        if sum(isnan(final_align_face(:))) ~= 0; keyboard; end   
        if min(final_align_face(:)) ~= 0; keyboard; end   
        if max(final_align_face(:)) ~= 1; keyboard; end   
        if numel(final_align_face) ~= resize_w_h*resize_w_h; keyboard; end   
        
        func_fig2_finalSelect_MaxConfidenceResult
        
        saveas(gcf,[saveFolder,'\',...,
                    'fig_',dataType_str,'_s',num2str(sample_idx),'_',...,
                    orig_title(1:end-4),'_2_case',case_flag_str,'.jpg'],'jpg');
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%     
    
    %% 3/3: Save & Clear   
    disp('---------------------------------------------------------------')
    disp(['Save: ',dataType_str,' s', num2str(sample_idx),'/',num2str(size(origImg_collec,1))])
    disp(' ')     
    
    save([saveFolder,'\',...,
            'AllPipelines_',dataType_str,'_s',num2str(sample_idx),'.mat'],...,
            'face_loc_1', 'face_im_1', 'faceCandi_set_1', 'end_flag_fd_1',...,
                'output_1', 'outFace_align_1', 'outLandmark_1', 'outFace_reCrop_1', 'align_info_1',...,
                'lmk_conf_1', 'end_flag_lmk_1', 'time_fd_1', 'time_lmk_1',...,
            'face_loc_2', 'face_im_2', 'faceCandi_set_2', 'end_flag_fd_2',...,
                'output_2', 'outFace_align_2', 'outLandmark_2', 'outFace_reCrop_2', 'align_info_2',...,
                'lmk_conf_2', 'end_flag_lmk_2', 'time_fd_2', 'time_lmk_2',...,            
            'face_loc_3', 'face_im_3', 'bs_3', 'end_flag_fd_3',...,
                'output_3', 'outFace_align_3', 'outLandmark_3', 'outFace_reCrop_3', 'align_info_3',...,
                'lmk_conf_3', 'end_flag_lmk_3', 'time_fd_3', 'time_lmk_3',...,            
            'face_loc_4', 'face_im_4', 'bs_4', 'end_flag_fd_4',...,
                'output_4', 'outFace_align_4', 'outLandmark_4', 'outFace_reCrop_4', 'align_info_4',...,
                'lmk_conf_4', 'end_flag_lmk_4', 'time_fd_4', 'time_lmk_4',...,
            'total_time_eachSample','orig_title');          
    
    save([saveFolder,'\',...,
            'finalSelect_',dataType_str,'_s',num2str(sample_idx),'.mat'],...,
          'case_flag_str', 'select_pipeline_idx', 'select_pipeline_str', ...,
          'final_align_face', 'final_outLandmark', 'final_outFace_reCrop', 'final_align_info',...,
          'collec_lmk_flag', 'collec_lmk_conf', 'collec_Fd_LMKd_result','orig_title');

    close all     

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
    clear face_loc_1 face_im_1 faceCandi_set_1 end_flag_fd_1
    clear output_1 outFace_align_1 outLandmark_1 outFace_reCrop_1 align_info_1
    clear lmk_conf_1 end_flag_lmk_1 time_fd_1 time_lmk_1
    
    clear face_loc_2 face_im_2 faceCandi_set_2 end_flag_fd_2
    clear output_2 outFace_align_2 outLandmark_2 outFace_reCrop_2 align_info_2
    clear lmk_conf_2 end_flag_lmk_2 time_fd_2 time_lmk_2   
    
    clear face_loc_3 face_im_3 bs_3 end_flag_fd_3
    clear output_3 outFace_align_3 outLandmark_3 outFace_reCrop_3 align_info_3
    clear lmk_conf_3 end_flag_lmk_3 time_fd_3 time_lmk_3
    
    clear face_loc_4 face_im_4 bs_4 end_flag_fd_4
    clear output_4 outFace_align_4 outLandmark_4 outFace_reCrop_4 align_info_4
    clear lmk_conf_4 end_flag_lmk_4 time_fd_4 time_lmk_4    
    
    clear raw_img_color raw_img_gray raw_img_histeq orig_title total_time_eachSample
    
end

time_process = toc(time_process_0);
disp(['  processing time = ',num2str(time_process,3),' sec'])
disp(' ')