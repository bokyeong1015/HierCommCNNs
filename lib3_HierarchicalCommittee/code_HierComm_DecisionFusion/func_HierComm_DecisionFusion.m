function outputData = func_HierComm_DecisionFusion(flag_CommLevel, fileList, gen_info)

    %% Loading data (level 1) or Collection of outputs of previous levels (level 2,3,...) 
    if flag_CommLevel == 1
        
        for m_idx = 1:1:size(fileList,1);
            tempFileName = fileList{m_idx, 1};       
            load(tempFileName,'collec_mean_scores','esti_label');
            
            inputData.esti_score(:,:,m_idx) = collec_mean_scores; % numClass x numSample x numModel
            inputData.esti_label(m_idx,:) = esti_label; % numModel x numSample
            clear esti_label collec_mean_scores
            
            if m_idx == size(fileList,1)
                load(tempFileName, 'query_label');
                inputData.true_label = query_label; clear query_label
            end            
            
            if gen_info.vote_flag(flag_CommLevel) == 4 ||... % VA-simp-WA rule
                    gen_info.vote_flag(flag_CommLevel) == 5; % VA-expo-WA rule
                %%%%%%%%%%%%%%%%%%% load valid info %%%%%%%%%%%%%%%%%%%
                tempFileName_valid = fileList{m_idx,2};
                load(tempFileName_valid,'acc_unbal','collec_mean_scores');
                
                inputData.valid_info.acc(m_idx, 1) = acc_unbal;
                inputData.valid_info.score(:,:,m_idx) = collec_mean_scores;     
                
                if m_idx == size(fileList,1)
                    load(tempFileName_valid,'query_label');      
                    inputData.valid_info.true_label = query_label; clear temp_query_label
                end                   
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            end
        end
        
    else
        
        inputData.esti_label = [];
        inputData.esti_score = [];
        
        for m_idx = 1:1:size(fileList,1);
            temp_outputData = fileList{m_idx, 1};
            inputData.esti_label = cat(1, inputData.esti_label, temp_outputData.esti_label);
            inputData.esti_score = cat(3, inputData.esti_score, temp_outputData.esti_score);
            
            if m_idx == size(fileList,1)
                inputData.true_label = temp_outputData.true_label;
            end
        end
        
    end
    
    %% Decision Fusion
    fprintf(['CommLev ',num2str(flag_CommLevel),' ',num2str(size(fileList,1)),' models'])
    
    if gen_info.vote_flag(flag_CommLevel) == 1; % majority vote of labels
        outputData = func_DecFusion_SimpMajVote(inputData, 0);        
        fprintf([': 1) SimpMajorVote, Acc = ',num2str(outputData.acc_unbal, 4),' \n']);
        
    elseif gen_info.vote_flag(flag_CommLevel) == 2; % median of probability vectors
        outputData = func_DecFusion_SimpMedian(inputData);
        fprintf([': 2) SimpMedi, Acc = ',num2str(outputData.acc_unbal, 4),' \n']);
        
    elseif gen_info.vote_flag(flag_CommLevel) == 3; % simple ave of probability vectors 
        outputData = func_DecFusion_SimpAve(inputData);
        fprintf([': 3) SimpAve, Acc = ',num2str(outputData.acc_unbal, 4),' \n']);
        
    elseif gen_info.vote_flag(flag_CommLevel) == 4; % VA-simp-WA rule
        outputData = func_DecFusion_VAsimpWA(inputData);
        fprintf([': 4) VAsimpWA, Acc = ',num2str(outputData.acc_unbal, 4),...,
            ' | final q = ',num2str(outputData.info_VAsimpWA.final_para),' \n']);
        
    elseif gen_info.vote_flag(flag_CommLevel) == 5; % VA-expo-WA rule
        outputData = func_DecFusion_VAexpoWA(inputData);
        fprintf([': 5) VAexpoWA, Acc = ',num2str(outputData.acc_unbal, 4),...,
            ' | final q = ',num2str(outputData.info_VAexpoWA.final_para),' \n']);
    end

end