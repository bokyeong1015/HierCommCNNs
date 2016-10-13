clc; clear; close all

%%% This example code gives how our hierarchical committees of deep CNNs are formed 
%%% and how they work with various decision fusion rules.

%%% However, for a complete code used in our article, you need to
%%% 1) train all 240 deep CNNs specified in our article using codes in lib2_TrainDeepCNN
%%% 2) collec posterior class probabilities yielded from the deep CNNs of '1)'
%%% 3) modify this code according to the hierarchy configuration in our article
%%% Here, we do not provide the trained models and the posterior probabilities due to
%%% the memory issue for uploading. Please carefully read our article and modify this code properly. 

%% General Setting
addpath code_HierComm_DecisionFusion

loadData_folder = 'sampleData_SFEW2.0_valid';
%%% This folder contains some sample data for hieararchical committee
%%% : posterior class probabilities of SFEW2.0 validation data 
%%%   yielded from 36 deep CNNs
%%% 36 deep CNNs = 1 (our aligned faces, oA) x 3 (input normalization, i.e., raw, iNor, cEnh) 
%%%                x 3 (size of CNN receptive fields, i.e., small, medium, large) 
%%%                x 4 (# neurons in the fc layer, i.e., 3072, 2048, 1024, 512) x 1 (RNG number 1) 

%%% each file (e.g., feat_valid_oA_DATAraw_CNNmed_FC3072_RNG1) includes
%%% 'collec_mean_scores': posterior class probabilities, numClass x numSample
%%% 'esti_label': estimated class labels from the corresponding deep CNN, 1 x numSample
%%% 'query_label': true class labels, 1 x numSample
%%% 'acc_unbal': accuracy (%), 100 x (# correctly estimated samples / # all samples)

data_type = 'valid'; % query data type, e.g., train, valid, test

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Decision fusion rule for each level in committee
% We set the 1st level rule as the VA-expo-WA rule
% and vary the higher level rules as the majority vote, median, and simple average rules
gen_info_HierComm.vote_flag(1) = 5; % VA-expo-WA rule

% gen_info.vote_flag(flag_CommLevel) == 1; % majority vote of labels
% gen_info.vote_flag(flag_CommLevel) == 2; % median of probability vectors 
% gen_info.vote_flag(flag_CommLevel) == 3; % simple ave of probability vectors 
% gen_info.vote_flag(flag_CommLevel) == 4; % VA-simp-WA rule
% gen_info.vote_flag(flag_CommLevel) == 5; % VA-expo-WA rule


%% Committee level 1
flag_CommLevel = 1;  
count_idx_CommLev1 = 1;
  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Committee level 1 - group G1
% collec 12 files of posterior probabilities
% which are obtained using 'raw' images for training deep CNNs

flag_input = 'raw'; rngNum = 1;

temp_count_idx_file = 1;
for cnn_idx = 1:1:3
    
    if cnn_idx == 1; flag_cnn = 'sma'; 
    elseif cnn_idx == 2; flag_cnn = 'med';
    elseif cnn_idx == 3; flag_cnn = 'lar';
    end    
    
    for fc_idx = 1:1:4
        if fc_idx == 1; num_fcHiddenNeuron = 3072;
        elseif fc_idx == 2; num_fcHiddenNeuron = 2048;
        elseif fc_idx == 3; num_fcHiddenNeuron = 1024;
        elseif fc_idx == 0; num_fcHiddenNeuron = 512;
        end            
        
        temp_loadFileName_query = ['feat_',data_type,'_oA_DATA',flag_input,'_CNN',flag_cnn,...,
                                    '_FC',num2str(num_fcHiddenNeuron),'_RNG',num2str(rngNum)];
        temp_loadFileName_valid = ['feat_valid_oA_DATA',flag_input,'_CNN',flag_cnn,...,
                                    '_FC',num2str(num_fcHiddenNeuron),'_RNG',num2str(rngNum)];                                
                                
        fileList{temp_count_idx_file,1} = [loadData_folder,'\',temp_loadFileName_query];  
        fileList{temp_count_idx_file,2} = [loadData_folder,'\',temp_loadFileName_valid];  

        temp_count_idx_file = temp_count_idx_file + 1;     
    end
end

outputData_lev1{count_idx_CommLev1,1} = func_HierComm_DecisionFusion(flag_CommLevel, fileList, gen_info_HierComm);
count_idx_CommLev1 = count_idx_CommLev1 + 1;
clear temp_count_idx fileList

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Committee level 1 - group G2
% collec 12 files of posterior probabilities
% which are obtained using 'iNor' images for training deep CNNs

flag_input = 'iNor'; rngNum = 1;

temp_count_idx_file = 1;
for cnn_idx = 1:1:3
    
    if cnn_idx == 1; flag_cnn = 'sma'; 
    elseif cnn_idx == 2; flag_cnn = 'med';
    elseif cnn_idx == 3; flag_cnn = 'lar';
    end    
    
    for fc_idx = 1:1:4
        if fc_idx == 1; num_fcHiddenNeuron = 3072;
        elseif fc_idx == 2; num_fcHiddenNeuron = 2048;
        elseif fc_idx == 3; num_fcHiddenNeuron = 1024;
        elseif fc_idx == 0; num_fcHiddenNeuron = 512;
        end            
        
        temp_loadFileName_query = ['feat_',data_type,'_oA_DATA',flag_input,'_CNN',flag_cnn,...,
                                    '_FC',num2str(num_fcHiddenNeuron),'_RNG',num2str(rngNum)];
        temp_loadFileName_valid = ['feat_valid_oA_DATA',flag_input,'_CNN',flag_cnn,...,
                                    '_FC',num2str(num_fcHiddenNeuron),'_RNG',num2str(rngNum)];                                
                                
        fileList{temp_count_idx_file,1} = [loadData_folder,'\',temp_loadFileName_query];  
        fileList{temp_count_idx_file,2} = [loadData_folder,'\',temp_loadFileName_valid];  

        temp_count_idx_file = temp_count_idx_file + 1;     
    end
end

outputData_lev1{count_idx_CommLev1,1} = func_HierComm_DecisionFusion(flag_CommLevel, fileList, gen_info_HierComm);
count_idx_CommLev1 = count_idx_CommLev1 + 1;
clear temp_count_idx fileList


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Committee level 1 - group G3
% collec 12 files of posterior probabilities
% which are obtained using 'cEnh' images for training deep CNNs

flag_input = 'cEnh'; rngNum = 1;

temp_count_idx_file = 1;
for cnn_idx = 1:1:3
    
    if cnn_idx == 1; flag_cnn = 'sma'; 
    elseif cnn_idx == 2; flag_cnn = 'med';
    elseif cnn_idx == 3; flag_cnn = 'lar';
    end    
    
    for fc_idx = 1:1:4
        if fc_idx == 1; num_fcHiddenNeuron = 3072;
        elseif fc_idx == 2; num_fcHiddenNeuron = 2048;
        elseif fc_idx == 3; num_fcHiddenNeuron = 1024;
        elseif fc_idx == 0; num_fcHiddenNeuron = 512;
        end            
        
        temp_loadFileName_query = ['feat_',data_type,'_oA_DATA',flag_input,'_CNN',flag_cnn,...,
                                    '_FC',num2str(num_fcHiddenNeuron),'_RNG',num2str(rngNum)];
        temp_loadFileName_valid = ['feat_valid_oA_DATA',flag_input,'_CNN',flag_cnn,...,
                                    '_FC',num2str(num_fcHiddenNeuron),'_RNG',num2str(rngNum)];                                
                                
        fileList{temp_count_idx_file,1} = [loadData_folder,'\',temp_loadFileName_query];  
        fileList{temp_count_idx_file,2} = [loadData_folder,'\',temp_loadFileName_valid];  

        temp_count_idx_file = temp_count_idx_file + 1;     
    end
end

outputData_lev1{count_idx_CommLev1,1} = func_HierComm_DecisionFusion(flag_CommLevel, fileList, gen_info_HierComm);
count_idx_CommLev1 = count_idx_CommLev1 + 1;
clear temp_count_idx fileList


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Committee level 1 - group G4
% collec 12 files of posterior probabilities
% which are obtained from the deep CNNs having 'small' size of receptive fields

flag_cnn = 'sma'; rngNum = 1;

temp_count_idx_file = 1;
for input_idx = 1:1:3
    
    if input_idx == 1; flag_input = 'raw'; 
    elseif input_idx == 2; flag_input = 'iNor';
    elseif input_idx == 3; flag_input = 'cEnh';
    end    
    
    for fc_idx = 1:1:4
        if fc_idx == 1; num_fcHiddenNeuron = 3072;
        elseif fc_idx == 2; num_fcHiddenNeuron = 2048;
        elseif fc_idx == 3; num_fcHiddenNeuron = 1024;
        elseif fc_idx == 0; num_fcHiddenNeuron = 512;
        end            
        
        temp_loadFileName_query = ['feat_',data_type,'_oA_DATA',flag_input,'_CNN',flag_cnn,...,
                                    '_FC',num2str(num_fcHiddenNeuron),'_RNG',num2str(rngNum)];
        temp_loadFileName_valid = ['feat_valid_oA_DATA',flag_input,'_CNN',flag_cnn,...,
                                    '_FC',num2str(num_fcHiddenNeuron),'_RNG',num2str(rngNum)];                                
                                
        fileList{temp_count_idx_file,1} = [loadData_folder,'\',temp_loadFileName_query];  
        fileList{temp_count_idx_file,2} = [loadData_folder,'\',temp_loadFileName_valid];  

        temp_count_idx_file = temp_count_idx_file + 1;     
    end
end

outputData_lev1{count_idx_CommLev1,1} = func_HierComm_DecisionFusion(flag_CommLevel, fileList, gen_info_HierComm);
count_idx_CommLev1 = count_idx_CommLev1 + 1;
clear temp_count_idx fileList


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Committee level 1 - group G5
% collec 12 files of posterior probabilities
% which are obtained from the deep CNNs having 'medium' size of receptive fields

flag_cnn = 'med'; rngNum = 1;

temp_count_idx_file = 1;
for input_idx = 1:1:3
    
    if input_idx == 1; flag_input = 'raw'; 
    elseif input_idx == 2; flag_input = 'iNor';
    elseif input_idx == 3; flag_input = 'cEnh';
    end    
    
    for fc_idx = 1:1:4
        if fc_idx == 1; num_fcHiddenNeuron = 3072;
        elseif fc_idx == 2; num_fcHiddenNeuron = 2048;
        elseif fc_idx == 3; num_fcHiddenNeuron = 1024;
        elseif fc_idx == 0; num_fcHiddenNeuron = 512;
        end            
        
        temp_loadFileName_query = ['feat_',data_type,'_oA_DATA',flag_input,'_CNN',flag_cnn,...,
                                    '_FC',num2str(num_fcHiddenNeuron),'_RNG',num2str(rngNum)];
        temp_loadFileName_valid = ['feat_valid_oA_DATA',flag_input,'_CNN',flag_cnn,...,
                                    '_FC',num2str(num_fcHiddenNeuron),'_RNG',num2str(rngNum)];                                
                                
        fileList{temp_count_idx_file,1} = [loadData_folder,'\',temp_loadFileName_query];  
        fileList{temp_count_idx_file,2} = [loadData_folder,'\',temp_loadFileName_valid];  

        temp_count_idx_file = temp_count_idx_file + 1;     
    end
end

outputData_lev1{count_idx_CommLev1,1} = func_HierComm_DecisionFusion(flag_CommLevel, fileList, gen_info_HierComm);
count_idx_CommLev1 = count_idx_CommLev1 + 1;
clear temp_count_idx fileList


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Committee level 1 - group G6
% collec 12 files of posterior probabilities
% which are obtained from the deep CNNs having 'medium' size of receptive fields

flag_cnn = 'lar'; rngNum = 1;

temp_count_idx_file = 1;
for input_idx = 1:1:3
    
    if input_idx == 1; flag_input = 'raw'; 
    elseif input_idx == 2; flag_input = 'iNor';
    elseif input_idx == 3; flag_input = 'cEnh';
    end    
    
    for fc_idx = 1:1:4
        if fc_idx == 1; num_fcHiddenNeuron = 3072;
        elseif fc_idx == 2; num_fcHiddenNeuron = 2048;
        elseif fc_idx == 3; num_fcHiddenNeuron = 1024;
        elseif fc_idx == 0; num_fcHiddenNeuron = 512;
        end            
        
        temp_loadFileName_query = ['feat_',data_type,'_oA_DATA',flag_input,'_CNN',flag_cnn,...,
                                    '_FC',num2str(num_fcHiddenNeuron),'_RNG',num2str(rngNum)];
        temp_loadFileName_valid = ['feat_valid_oA_DATA',flag_input,'_CNN',flag_cnn,...,
                                    '_FC',num2str(num_fcHiddenNeuron),'_RNG',num2str(rngNum)];                                
                                
        fileList{temp_count_idx_file,1} = [loadData_folder,'\',temp_loadFileName_query];  
        fileList{temp_count_idx_file,2} = [loadData_folder,'\',temp_loadFileName_valid];  

        temp_count_idx_file = temp_count_idx_file + 1;     
    end
end

outputData_lev1{count_idx_CommLev1,1} = func_HierComm_DecisionFusion(flag_CommLevel, fileList, gen_info_HierComm);
count_idx_CommLev1 = count_idx_CommLev1 + 1;
clear temp_count_idx fileList

%% Committee level 2
disp(' ')
disp('================================================================');

flag_CommLevel = 2;  
count_idx_CommLev2 = 1;

for vote_flag = 1:1:3
    gen_info_HierComm.vote_flag(flag_CommLevel) = vote_flag;        
    outputData_lev2 = ...
        func_HierComm_DecisionFusion(flag_CommLevel, outputData_lev1, gen_info_HierComm);

    resultCollec{vote_flag, 1} = outputData_lev2.acc_unbal;
    % final accuracy for each fusion rule in the second level 
    
    resultCollec{vote_flag, 2} = outputData_lev2.esti_label; 
    % final estimated label for each fusion rule in the second level 

    clear outputData_lev2
end % end for vote_flag



