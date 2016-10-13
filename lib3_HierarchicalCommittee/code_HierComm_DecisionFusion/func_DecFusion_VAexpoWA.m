function outputData = func_DecFusion_VAexpoWA(inputData)
   
    input_esti_score = inputData.esti_score; % input_esti_score: numClass x numSample x numModel
    true_label = inputData.true_label; % 1 x numSample    
   
    valid_info = inputData.valid_info;
    % valid_info.score: numClass x numSample x numModel
    % valid_info.true_label: 1 x numSample
    % valid_info.acc: numModel x 1
    
    
    %% exponent selection
    para_arr = [-50:0.1:150];
    num_totalExp = size(valid_info.score, 3);    
    
%     fprintf(['  ',num2str(num_totalExp),' model VAexpoWA... ']);

    for para_idx = 1:1:size(para_arr,2)
        temp_para = para_arr(para_idx);
%         disp('===============================================================================')
%         disp([num2str(num_totalExp),' model Ave: ', num2str(para_idx),'/',num2str(size(para_arr,2)),...,
%             ' para = ',num2str(temp_para)]);

        temp_w0 = valid_info.acc .^ temp_para; % numModel x 1
        temp_decWeight = temp_w0./sum(temp_w0);

        for temp_exp_idx = 1:1:num_totalExp
            temp_weight = temp_decWeight(temp_exp_idx,1);
            temp_decWeight_mask(:,:,temp_exp_idx) = temp_weight .*...,
                ones(size(valid_info.score, 1), size(valid_info.score, 2));
            clear temp_weight
        end

        temp_VAexpoWA_z = double(valid_info.score) .* temp_decWeight_mask; % numClass x numSample x numModel        
        tempFinal_VAexpoWA_z = sum(temp_VAexpoWA_z, 3);

        [~, temp_output_esti_label_valid] = max(tempFinal_VAexpoWA_z);
        collecScan_accValid(para_idx, 1) = 100 * sum(valid_info.true_label == temp_output_esti_label_valid)/length(valid_info.true_label);      
        clear temp_w0 temp_decWeight temp_decWeight_mask temp_VAexpoWA_z tempFinal_VAexpoWA_z temp_output_esti_label_valid
    end    
    
    [~, para_select_idx] = max(collecScan_accValid);
    final_para = para_arr(para_select_idx);
    
%     fprintf([' -> final exponent = ',num2str(final_para),' \n']);
    
    %% applying the selected exponent to query data
    w0 = valid_info.acc .^ final_para; % numModel x 1
    decWeight = w0./sum(w0); clear w0
    
    for temp_exp_idx = 1:1:num_totalExp
        temp_weight = decWeight(temp_exp_idx,1);
        decWeight_mask(:,:,temp_exp_idx) = temp_weight .*...,
            ones(size(input_esti_score, 1), size(input_esti_score, 2));
        clear temp_weight
    end    
    
    temp_VAexpoWA_z = double(input_esti_score) .* decWeight_mask; % numClass x numSample x numModel        
    final_VAexpoWA_z = sum(temp_VAexpoWA_z, 3);

    [~, output_esti_label] = max(final_VAexpoWA_z);
    acc_unbal = 100 * sum(true_label == output_esti_label)/length(true_label);
    
    info_VAexpoWA.decWeight = decWeight;
    info_VAexpoWA.final_para = final_para;
    info_VAexpoWA.orig_valAcc = valid_info.acc;
    
    outputData.esti_label = output_esti_label;
    outputData.esti_score = final_VAexpoWA_z;
    outputData.true_label = true_label;
    outputData.acc_unbal = 100 * sum(true_label == output_esti_label)/length(true_label);
    outputData.info_VAexpoWA = info_VAexpoWA;
    
end
