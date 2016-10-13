%% 1. 'VjFd-Raw'
prep_str_1 = '1. Color IMG - VJ Model - Gray IMG - IF Model';
disp(['s',num2str(sample_idx),': ',prep_str_1]);

%%%%%%%%%%% Face Detect
fprintf(['  FaceDetect... '])
time_fd = tic;

[face_loc_1, face_im_1, faceCandi_set_1, end_flag_fd_1] =...
    func_FaceDetection_VJmodel(raw_img_color, module_VJmodel_path, fig_flag_faceDetect_1);

time_fd_1 = toc(time_fd);
fprintf([num2str(time_fd_1,3),' sec \n']);

%%%%%%%%%%% Landmark Detect
fprintf(['  LmkDetect + Align... '])
time_lmk = tic;

if end_flag_fd_1 == 1
    output_1 = xx_track_detect(Models, raw_img_gray, face_loc_1, model_option);
    if ~isempty(output_1.pred) % FaceDetection O - LandmarkDetection O
        end_flag_lmk_1 = 1;
        [outFace_align_1, outLandmark_1, outFace_reCrop_1, align_info_1] = ...    
                    func_FaceAlign_RuleBasedRotateCrop(raw_img_gray, output_1,...,
                                                          resize_w_h, face_loc_1, fig_flag_lmkDetect_1);
        lmk_conf_1 = output_1.conf;
    else % FaceDetection O - LandmarkDetection X
        outFace_align_1 = -1;            
        outLandmark_1 = zeros(49,2);
        outFace_reCrop_1 = -1;
        align_info_1 = [];

        end_flag_lmk_1 = -1;
        lmk_conf_1 = -1;
    end
else % FaceDetection X - LandmarkDetection X
    face_loc_1 = [0 0 0 0];
    output_1 = [];
    outFace_align_1 = -2;
    outLandmark_1 = zeros(49,2);
    outFace_reCrop_1 = -2;
    align_info_1 = [];   

    end_flag_lmk_1 = -2;
    lmk_conf_1 = -2;    
end

time_lmk_1 = toc(time_lmk);
fprintf([num2str(time_lmk_1,3),' sec \n']);

%% 2. 'VjFd-HistEq'
prep_str_2 = '2. Gray/HistEq IMG - VJ Model - Gray/HistEq IMG - IF Model';
disp(['s',num2str(sample_idx),': ',prep_str_2]);

%%%%%%%%%%% Face Detect
fprintf(['  FaceDetect... '])
time_fd = tic;

[face_loc_2, face_im_2, faceCandi_set_2, end_flag_fd_2] =...
    func_FaceDetection_VJmodel(raw_img_histeq, module_VJmodel_path, fig_flag_faceDetect_2);

time_fd_2 = toc(time_fd);
fprintf([num2str(time_fd_2,3),' sec \n']);

%%%%%%%%%%% Landmark Detect
fprintf(['  LmkDetect + Align... '])
time_lmk = tic;

if end_flag_fd_2 == 1
    output_2 = xx_track_detect(Models, raw_img_histeq, face_loc_2, model_option);
    if ~isempty(output_2.pred) % FaceDetection O - LandmarkDetection O
        end_flag_lmk_2 = 1;
        [outFace_align_2, outLandmark_2, outFace_reCrop_2, align_info_2] = ...    
                    func_FaceAlign_RuleBasedRotateCrop(raw_img_histeq, output_2,...,
                                                          resize_w_h, face_loc_2, fig_flag_lmkDetect_2);
        lmk_conf_2 = output_2.conf;
    else % FaceDetection O - LandmarkDetection X
        outFace_align_2 = -1;
        outLandmark_2 = zeros(49,2);
        outFace_reCrop_2 = -1;
        align_info_2 = [];

        end_flag_lmk_2 = -1;
        lmk_conf_2 = -1;
    end
else % FaceDetection X - LandmarkDetection X
    face_loc_2 = [0 0 0 0];
    output_2 = [];
    outFace_align_2 = -2;
    outLandmark_2 = zeros(49,2);
    outFace_reCrop_2 = -2;
    align_info_2 = [];   

    end_flag_lmk_2 = -2;
    lmk_conf_2 = -2;    
end

time_lmk_2 = toc(time_lmk);
fprintf([num2str(time_lmk_2,3),' sec \n']);

%% 3. 'XZhuFd-Raw'
prep_str_3 = '3. Color IMG - ZR Model - Gray IMG - IF Model';
disp(['s',num2str(sample_idx),': ',prep_str_3]);

%%%%%%%%%%% Face Detect
fprintf(['  FaceDetect... '])
time_fd = tic;    

fprintf(['1..']);
bs_3 = detect(raw_img_color, model, model.thresh);
fprintf(['2..']);
bs_3 = clipboxes(raw_img_color, bs_3);
fprintf(['3.. ']);
bs_3 = nms_face(bs_3, 0.3);

if isempty(bs_3) ~= 1
    end_flag_fd_3 = 1;

    temp_bs = bs_3(1).xy;
    bs_3_s = bs_3(1).s;
    bs_3_xy(:,1) = (temp_bs(:,1)+temp_bs(:,3))/2;
    bs_3_xy(:,2) = (temp_bs(:,2)+temp_bs(:,4))/2;    
    clear temp_bs temp_s

    temp_xx_min = min(bs_3_xy(:,1)); temp_xx_max = max(bs_3_xy(:,1));
    temp_yy_min = min(bs_3_xy(:,2)); temp_yy_max = max(bs_3_xy(:,2));

    temp_del_xx = abs((temp_xx_max - temp_xx_min))/10;
    temp_del_yy = abs((temp_yy_max - temp_yy_min))/10;

    temp_xx_min = round(temp_xx_min - temp_del_xx);
    temp_xx_max = round(temp_xx_max + temp_del_xx);
    temp_yy_min = round(temp_yy_min - temp_del_yy);
    temp_yy_max = round(temp_yy_max + temp_del_yy);

    if temp_xx_min < 1; temp_xx_min = 1; end
    if temp_yy_min < 1; temp_yy_min = 1; end
    if temp_xx_max > size(raw_img_gray,2); temp_xx_max = size(raw_img_gray,2); end
    if temp_yy_max > size(raw_img_gray,1); temp_yy_max = size(raw_img_gray,1); end    

    face_loc_3 = [temp_xx_min, temp_yy_min,...,
        temp_xx_max - temp_xx_min + 1, temp_yy_max - temp_yy_min + 1];
    face_im_3 = raw_img_color(temp_yy_min:temp_yy_max, temp_xx_min:temp_xx_max, :);

    clear temp_xx_min temp_yy_min temp_xx_max temp_yy_max
    clear temp_del_xx temp_del_yy    
    clear bs_3_xy bs_3_s
else
    end_flag_fd_3 = -1;
    face_im_3 = -1;
end

time_fd_3 = toc(time_fd);
fprintf([num2str(time_fd_3,3),' sec \n']);

%%%%%%%%%%% Landmark Detect
fprintf(['  LmkDetect + Align... '])
time_lmk = tic;

if end_flag_fd_3 == 1
    output_3 = xx_track_detect(Models, raw_img_gray, face_loc_3, model_option);
    if ~isempty(output_3.pred) % FaceDetection O - LandmarkDetection O
        end_flag_lmk_3 = 1;
        [outFace_align_3, outLandmark_3, outFace_reCrop_3, align_info_3] = ...    
                    func_FaceAlign_RuleBasedRotateCrop(raw_img_gray, output_3,...,
                                                          resize_w_h, face_loc_3, fig_flag_lmkDetect_3);
        lmk_conf_3 = output_3.conf;
    else % FaceDetection O - LandmarkDetection X
        outFace_align_3 = -1;
        outLandmark_3 = zeros(49,2);
        outFace_reCrop_3 = -1;
        align_info_3 = [];

        end_flag_lmk_3 = -1;
        lmk_conf_3 = -1;
    end
else % FaceDetection X - LandmarkDetection X
    face_loc_3 = [0 0 0 0];
    output_3 = [];
    outFace_align_3 = -2;
    outLandmark_3 = zeros(49,2);
    outFace_reCrop_3 = -2;
    align_info_3 = [];   

    end_flag_lmk_3 = -2;
    lmk_conf_3 = -2;    
end

time_lmk_3 = toc(time_lmk);
fprintf([num2str(time_lmk_3,3),' sec \n']);

%% 4. 'XZhuFd-HistEq'
prep_str_4 = '4. Color IMG - ZR Model - Gray/HistEq IMG - IF Model';
disp(['s',num2str(sample_idx),': ',prep_str_4]);

%%%%%%%%%%% Face Detect
fprintf(['  FaceDetect... '])
time_fd = tic;

bs_4 = bs_3;
end_flag_fd_4 = end_flag_fd_3;

if end_flag_fd_4 == 1
    face_loc_4 = face_loc_3;
    face_im_4 = face_im_3;
else
    face_loc_4 = [0 0 0 0];
    face_im_4 = -1;
end

time_fd_4 = toc(time_fd);
fprintf([num2str(time_fd_4,3),' sec \n']);

%%%%%%%%%%% Landmark Detect
fprintf(['  LmkDetect + Align... '])
time_lmk = tic;

if end_flag_fd_4 == 1
    output_4 = xx_track_detect(Models, raw_img_histeq, face_loc_4, model_option);
    if ~isempty(output_4.pred) % FaceDetection O - LandmarkDetection O
        end_flag_lmk_4 = 1;
        [outFace_align_4, outLandmark_4, outFace_reCrop_4, align_info_4] = ...    
                    func_FaceAlign_RuleBasedRotateCrop(raw_img_histeq, output_4,...,
                                                          resize_w_h, face_loc_4, fig_flag_lmkDetect_4);
        lmk_conf_4 = output_4.conf;
    else % FaceDetection O - LandmarkDetection X
        outFace_align_4 = -1;
        outLandmark_4 = zeros(49,2);
        outFace_reCrop_4 = -1;
        align_info_4 = [];

        end_flag_lmk_4 = -1;
        lmk_conf_4 = -1;
    end
else % FaceDetection X - LandmarkDetection X
    face_loc_4 = [0 0 0 0];
    output_4 = [];
    outFace_align_4 = -2;
    outLandmark_4 = zeros(49,2);
    outFace_reCrop_4 = -2;
    align_info_4 = [];   

    end_flag_lmk_4 = -2;
    lmk_conf_4 = -2;    
end

time_lmk_4 = toc(time_lmk);
fprintf([num2str(time_lmk_4,3),' sec \n']);