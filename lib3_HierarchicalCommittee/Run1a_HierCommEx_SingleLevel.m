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
% Decision fusion rule for single-level committee

% gen_info.vote_flag(flag_CommLevel) == 1; % majority vote of labels
% gen_info.vote_flag(flag_CommLevel) == 2; % median of probability vectors 
% gen_info.vote_flag(flag_CommLevel) == 3; % simple ave of probability vectors 
% gen_info.vote_flag(flag_CommLevel) == 4; % VA-simp-WA rule
% gen_info.vote_flag(flag_CommLevel) == 5; % VA-expo-WA rule


%% Single-Level Committee
flag_CommLevel = 1;  

rngNum = 1;
temp_count_idx_file = 1;

% collec 36 files of posterior probabilities
for input_idx = 1:1:3
    
    if input_idx == 1; flag_input = 'raw'; 
    elseif input_idx == 2; flag_input = 'iNor';
    elseif input_idx == 3; flag_input = 'cEnh';
    end    

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

end

for vote_flag = 1:1:5
    gen_info_HierComm.vote_flag(flag_CommLevel) = vote_flag;        
    outputData = ...
        func_HierComm_DecisionFusion(flag_CommLevel, fileList, gen_info_HierComm);

    resultCollec{vote_flag, 1} = outputData.acc_unbal;
    % final accuracy for each fusion rule in the second level 
    
    resultCollec{vote_flag, 2} = outputData.esti_label; 
    % final estimated label for each fusion rule in the second level 

    clear outputData
end % end for vote_flag

