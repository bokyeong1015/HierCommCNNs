function outputData = func_DecFusion_SimpMajVote(inputData, flag_disp)
                                                                    
    input_esti_label = inputData.esti_label; % input_esti_label: numModel x numSample
    input_esti_score = inputData.esti_score; % input_esti_score: numClass x numSample x numModel
    true_label = inputData.true_label; % 1 x numSample
    
    numClass = size(input_esti_score, 1);
    
    [output_esti_label0, final_freq0] = mode(input_esti_label, 1); 
    % Most frequent values in array, Cor dir. -> 1 x numSample
    
    count_candi_sample = 0;
    for s_idx = 1:1:size(input_esti_label,2)
        temp_data = input_esti_label(:,s_idx);
        hist_label_count = histc(temp_data,[1:1:numClass]); % 1xnumClass, # vote of each class
        candi_class_set = find(hist_label_count == max(hist_label_count));

        if length(candi_class_set) == 1

            output_esti_label(1,s_idx) = candi_class_set;
            final_freq(1,s_idx) = hist_label_count(candi_class_set);

        elseif length(candi_class_set) > 1
            if flag_disp == 1
                disp(' ')
                disp(['   Sample ',num2str(s_idx),': same #votes.. select max class-prob.'])
            end
            
            temp_probSet_all = squeeze(input_esti_score(:, s_idx, :)); % numClass x numModel
            [~, temp_label] = max(temp_probSet_all) ; 

            for c_idx = 1:1:length(candi_class_set)
                candi_idx = candi_class_set(c_idx);
                temp_probSet_part = temp_probSet_all(:,temp_label == candi_idx); % numClass x numCandiModel
                temp_probSet_part_mean = mean(temp_probSet_part,2); % numClass x 1

                temp_collec_info(c_idx, 1) = candi_idx; % corresponding class
                temp_collec_info(c_idx, 2) = temp_probSet_part_mean(candi_idx); % mean class-probability
                temp_collec_info(c_idx, 3) = size(temp_probSet_part, 2);

                if flag_disp == 1
                    disp(['       class ',num2str(candi_idx),': prob ',num2str(temp_probSet_part_mean(candi_idx),3),...,
                        ' - by ',num2str(size(temp_probSet_part, 2)),' models'])
                end
                clear candi_idx temp_probSet_part temp_probSet_part_mean
            end
            [~, temp_select_idx] = max(temp_collec_info(:,2));

            %%%%%%%%%%%%%%%%
            output_esti_label(1,s_idx) = temp_collec_info(temp_select_idx, 1);
            final_freq(1,s_idx) = temp_collec_info(temp_select_idx, 3);      
            
            if flag_disp == 1
                disp(['   -> Select class ',num2str(output_esti_label(1,s_idx)),' where true class ',num2str(true_label(s_idx))]);
            end

            clear temp_probSet_all temp_label temp_collec_info temp_select_idx

            count_candi_sample = count_candi_sample + 1;
        end
        clear temp_data hist_label_count candi_class_set
    end
    
    if flag_disp == 1
        disp(' ');
        disp('-------------- Improvement -----------');
        disp([' ',num2str(count_candi_sample),' samples {Same # Votes -> Max Class-prob}']);
        disp([' Among them, ',num2str(sum(output_esti_label0~=output_esti_label)),...,
            '/',num2str(count_candi_sample),' different decisions !']);
    end
    
    outputData.esti_label = output_esti_label;
    outputData.esti_score = mean(input_esti_score, 3);
    outputData.true_label = true_label;
    outputData.acc_unbal = 100 * sum(true_label == output_esti_label)/length(true_label);

end
