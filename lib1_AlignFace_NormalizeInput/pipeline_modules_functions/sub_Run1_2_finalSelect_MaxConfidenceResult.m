collec_lmk_flag = [end_flag_lmk_1; end_flag_lmk_2; end_flag_lmk_3; end_flag_lmk_4];
collec_lmk_conf = [lmk_conf_1; lmk_conf_2; lmk_conf_3; lmk_conf_4];       

collec_Fd_LMKd_result{1,1} = output_1; collec_Fd_LMKd_result{1,2} = face_loc_1;
collec_Fd_LMKd_result{1,3} = '1. Color IMG - VJ Model - Gray IMG - IF Model';

collec_Fd_LMKd_result{2,1} = output_2; collec_Fd_LMKd_result{2,2} = face_loc_2;
collec_Fd_LMKd_result{2,3} = '2. Gray/HistEq IMG - VJ Model - Gray/HistEq IMG - IF Model';

collec_Fd_LMKd_result{3,1} = output_3; collec_Fd_LMKd_result{3,2} = face_loc_3;
collec_Fd_LMKd_result{3,3} = '3. Color IMG - ZR Model - Gray IMG - IF Model';
collec_Fd_LMKd_result{3,4} = bs_3;

collec_Fd_LMKd_result{4,1} = output_4; collec_Fd_LMKd_result{4,2} = face_loc_4;
collec_Fd_LMKd_result{4,3} = '4. Color IMG - ZR Model - Gray/HistEq IMG - IF Model';    
collec_Fd_LMKd_result{4,4} = bs_4;

[sortIdx_lmk_conf, sortValue_lmk_conf] = sort(collec_lmk_conf, 'descend');

if isempty(find(unique(collec_lmk_flag) == 1)) == 0 % at least one pipeline successes in both Fd & LMKd
    disp('Case 1. at least one pipeline successes in both Fd & LMKd');
    case_flag_str = '1';

    select_pipeline_idx = sortValue_lmk_conf(1);
    select_lmk_output = collec_Fd_LMKd_result{select_pipeline_idx,1};
    select_face_loc = collec_Fd_LMKd_result{select_pipeline_idx,2};
    select_pipeline_str = collec_Fd_LMKd_result{select_pipeline_idx,3};

    disp(['        final pipeline: ', select_pipeline_str])

    [final_outFace_align, final_outLandmark, final_outFace_reCrop, final_align_info] = ...    
                func_FaceAlign_RuleBasedRotateCrop(raw_img_gray, select_lmk_output,...,
                                                      resize_w_h, select_face_loc, 0);       
    temp_final_face = double(final_outFace_align);
    temp_final_face = imresize(temp_final_face, [resize_w_h resize_w_h]);
    min_value = min(temp_final_face(:)); 
    max_value = max(temp_final_face(:));

    final_align_face = (temp_final_face - min_value)./((max_value - min_value)+eps);    
    clear temp_final_face min_value max_value

else % all pipelines fail in both Fd & LMKd
    if isempty(find(unique(collec_lmk_flag) == -1)) == 0 % at least one pipeline successes in Fd
       disp('Case 2. at least one pipeline successes in Fd'); 
       candi_idx = find(collec_lmk_flag == -1);

       if length(candi_idx) == 1 % only one pipeline sucesses in Fd | must be 1 or 2
           disp(' -- Case 2a. only one pipeline sucesses in Fd | must be 1 or 2'); 
           case_flag_str = '2a';

           select_pipeline_idx = candi_idx;
           select_face_loc = collec_Fd_LMKd_result{select_pipeline_idx,2};
           select_pipeline_str = collec_Fd_LMKd_result{select_pipeline_idx,3};
           disp(['            final pipeline: ', select_pipeline_str])

           temp_x = select_face_loc(1); temp_y = select_face_loc(2);
           temp_w = select_face_loc(3); temp_h = select_face_loc(4);

           temp_final_face = double(raw_img_gray(temp_y:(temp_y+temp_h-1), temp_x:(temp_x+temp_w-1)));
           temp_final_face = imresize(temp_final_face, [resize_w_h resize_w_h]);
           min_value = min(temp_final_face(:)); 
           max_value = max(temp_final_face(:));

           final_align_face = (temp_final_face - min_value)./((max_value - min_value)+eps);   
           final_outLandmark = [];
           final_outFace_reCrop = [];
           final_align_info.facePos0_fdOnly = [temp_x, temp_y, temp_w, temp_h];     

           clear temp_final_face min_value max_value
           clear temp_x temp_y temp_w temp_h

       else % more than 2 pipelines success in Fd 
            % must be a pair (1,2), (3,4) or (1,3,4), (2,3,4) or (1,2,3,4)
           disp(' -- Case 2b. more than 2 pipelines success in Fd'); 
           disp('             must be a pair (1,2), (3,4) or (1,3,4), (2,3,4) or (1,2,3,4)')

            if  ~isempty(find(candi_idx == 3)) &&  ~isempty(find(candi_idx == 4))  %% Fd via 2 ZR models
                disp(' ---- Case 2b-1. Fd via 2 ZR models'); 
                disp('                 Retry LMKd based on tight face detection box')           

                select_pipeline_idx = 3;
                select_pipeline_str = collec_Fd_LMKd_result{select_pipeline_idx,3};
                disp(['                final pipeline: ', select_pipeline_str])

                % Note bs_3 = bs_4 
                bs_3_lmk(:,1) = (bs_3(1).xy(:,1) + bs_3(1).xy(:,3)) / 2;
                bs_3_lmk(:,2) = (bs_3(1).xy(:,2) + bs_3(1).xy(:,4)) / 2;            

                if size(bs_3_lmk,1) == 39; bs_lmk_new = bs_3_lmk(1:39-11,:); % 39 pts, last 11
                elseif size(bs_3_lmk,1) == 68; bs_lmk_new = bs_3_lmk(1:68-17,:); % 68 pts, last 17
                end

                temp_xx_min = min(bs_lmk_new(:,1)); temp_xx_max = max(bs_lmk_new(:,1));
                temp_yy_min = min(bs_lmk_new(:,2)); temp_yy_max = max(bs_lmk_new(:,2));

                temp_del_xx = abs((temp_xx_max - temp_xx_min))/10;
                temp_del_yy = abs((temp_yy_max - temp_yy_min))/10;

                temp_xx_min = round(temp_xx_min - temp_del_xx); temp_xx_max = round(temp_xx_max + temp_del_xx);
                temp_yy_min = round(temp_yy_min - temp_del_yy); temp_yy_max = round(temp_yy_max + temp_del_yy);

                if temp_xx_min < 1; temp_xx_min = 1; end
                if temp_yy_min < 1; temp_yy_min = 1; end
                if temp_xx_max > size(raw_img_gray,2); temp_xx_max = size(raw_img_gray,2); end
                if temp_yy_max > size(raw_img_gray,1); temp_yy_max = size(raw_img_gray,1); end    

                face_loc_new = [temp_xx_min, temp_yy_min,...,
                    temp_xx_max - temp_xx_min + 1, temp_yy_max - temp_yy_min + 1];

                output_new = xx_track_detect(Models, raw_img_gray, face_loc_new, model_option);       
                if ~isempty(output_new.pred) % FaceDetection O - LandmarkDetection O
                    disp(' ------ Case 2b-1a. Fd via 2 ZR models: Success in Retrying');                      
                    case_flag_str = '2b-1a';

                    if fig_flag_finalAlign == 1
                        [final_outFace_align, final_outLandmark, final_outFace_reCrop, final_align_info] = ...    
                                        func_FaceAlign_RuleBasedRotateCrop(raw_img_gray, output_new,...,
                                                                              resize_w_h, face_loc_new, 1); 
                        saveas(gcf,[saveFolder,'\',...,
                                    'fig_',dataType_str,'_s',num2str(sample_idx),'_',...,
                                    orig_title(1:end-4),'_3_retrySuccess.jpg'],'jpg');                                                                               
                    else
                        [final_outFace_align, final_outLandmark, final_outFace_reCrop, final_align_info] = ...    
                                        func_FaceAlign_RuleBasedRotateCrop(raw_img_gray, output_new,...,
                                                                              resize_w_h, face_loc_new, 0);                             
                    end

                    temp_final_face = double(final_outFace_align);
                    temp_final_face = imresize(temp_final_face, [resize_w_h resize_w_h]);
                    min_value = min(temp_final_face(:)); 
                    max_value = max(temp_final_face(:));

                    final_align_face = (temp_final_face - min_value)./((max_value - min_value)+eps);  
                    clear temp_final_face min_value max_value
                else     
                    disp(' ------ Case 2b-1b. Fd via 2 ZR models: Fail in Retrying');                      
                    case_flag_str = '2b-1b';

                    temp_final_face = double(raw_img_gray(temp_yy_min:temp_yy_max, temp_xx_min:temp_xx_max));
                    temp_final_face = imresize(temp_final_face, [resize_w_h resize_w_h]);
                    min_value = min(temp_final_face(:)); 
                    max_value = max(temp_final_face(:));

                    final_align_face = (temp_final_face - min_value)./((max_value - min_value)+eps); 
                    final_outLandmark = [];
                    final_outFace_reCrop = [];
                    final_align_info.facePos0_fdOnly = face_loc_new;                               
                    clear temp_final_face min_value max_value                     
                end   

                clear bs_3_lmk  bs_lmk_new face_loc_new output_new
                clear temp_xx_min temp_yy_min temp_xx_max temp_yy_max
                clear temp_del_xx temp_del_yy      

            elseif (~isempty(find(candi_idx == 1)) &&  ~isempty(find(candi_idx == 2))) &&...
                    (isempty(find(candi_idx == 3)) &&  isempty(find(candi_idx == 4))) 
                    %%%% Only when VJ model pairs work
                    disp(' ---- Case 2b-2. Fd via 2 VJ models'); 
                    case_flag_str = '2b-2';

                    select_pipeline_idx = 1;
                    select_pipeline_str = collec_Fd_LMKd_result{select_pipeline_idx,3};
                    disp(['                final pipeline: ', select_pipeline_str])

                    select_face_loc1 = collec_Fd_LMKd_result{candi_idx(1),2};
                    select_face_loc2 = collec_Fd_LMKd_result{candi_idx(2),2};

                    select_face_loc = (select_face_loc1 + select_face_loc2)/2;

                    temp_x = round(select_face_loc(1)); temp_y = round(select_face_loc(2));                   
                    temp_w = round(select_face_loc(3)); temp_h = round(select_face_loc(4));                   

                    temp_final_face = double(raw_img_gray(temp_y:(temp_y+temp_h-1), temp_x:(temp_x+temp_w-1)));
                    temp_final_face = imresize(temp_final_face, [resize_w_h resize_w_h]);
                    min_value = min(temp_final_face(:)); 
                    max_value = max(temp_final_face(:));

                    final_align_face = (temp_final_face - min_value)./((max_value - min_value)+eps); 
                    final_outLandmark = [];
                    final_outFace_reCrop = [];
                    final_align_info.facePos0_fdOnly = [temp_x, temp_y, temp_w, temp_h];          

                    clear temp_final_face min_value max_value         
                    clear temp_x temp_y temp_w temp_h
                    clear select_face_loc1 select_face_loc2
            end    
       end           
    else % all pipelines fail in Fd            
        disp('Case 3. all pipelines fail in Fd'); 
        case_flag_str = '3';

        select_pipeline_idx = 0;
        select_pipeline_str = 'None';
        disp(['                final pipeline: ', select_pipeline_str])

        [ran_y, ran_x] = size(raw_img_gray);            
        del_x = round(ran_x/3); del_y = round(ran_y/3);

        temp_x = del_x; temp_y = del_y;         
        temp_w = del_x; temp_h = del_y;  

        temp_final_face = double(raw_img_gray(temp_y:(temp_y+temp_h-1), temp_x:(temp_x+temp_w-1)));
        temp_final_face = imresize(temp_final_face, [resize_w_h resize_w_h]);
        min_value = min(temp_final_face(:)); 
        max_value = max(temp_final_face(:));

        final_align_face = (temp_final_face - min_value)./((max_value - min_value)+eps); 
        final_outLandmark = [];
        final_outFace_reCrop = [];
        final_align_info.facePos0_fdOnly = [temp_x, temp_y, temp_w, temp_h];  

        clear temp_final_face min_value max_value         
        clear temp_x temp_y temp_w temp_h       
    end        
end