function outputData = func_DecFusion_VAsimpWA(inputData)
   
    input_esti_score = inputData.esti_score; % input_esti_score: numClass x numSample x numModel
    true_label = inputData.true_label; % 1 x numSample    
   
    valid_info = inputData.valid_info;
    % valid_info.score: numClass x numSample x numModel
    % valid_info.true_label: 1 x numSample
    % valid_info.acc: numModel x 1

    num_totalExp = size(valid_info.score, 3);    
    final_para = 1;
    
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
    
    temp_VAsimpWA_z = double(input_esti_score) .* decWeight_mask; % numClass x numSample x numModel        
    final_VAsimpWA_z = sum(temp_VAsimpWA_z, 3);

    [~, output_esti_label] = max(final_VAsimpWA_z);
    acc_unbal = 100 * sum(true_label == output_esti_label)/length(true_label);
    
    info_VAsimpWA.decWeight = decWeight;
    info_VAsimpWA.final_para = final_para;
    info_VAsimpWA.orig_valAcc = valid_info.acc;
    
    outputData.esti_label = output_esti_label;
    outputData.esti_score = final_VAsimpWA_z;
    outputData.true_label = true_label;
    outputData.acc_unbal = 100 * sum(true_label == output_esti_label)/length(true_label);
    outputData.info_VAsimpWA = info_VAsimpWA;
    
end
