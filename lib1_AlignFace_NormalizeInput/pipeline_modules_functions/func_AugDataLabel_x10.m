function [data_withAug, label_withAug, set_withAug] = func_AugDataLabel_x10(data_orig, label_orig, set_orig, imSize_aug) 

numTimesAug = 10;
numOrigSample = size(data_orig, 4);
data_withAug = zeros(imSize_aug, imSize_aug, 1, numOrigSample*numTimesAug, 'single');
set_withAug = zeros(1, numOrigSample*numTimesAug);

if isempty(label_orig) ~= 1
    label_withAug = zeros(1, numOrigSample*numTimesAug);
else
    label_withAug = [];
end

numCount = 1;

for sample_idx = 1:1:numOrigSample
    disp(['Aug imdb',num2str(imSize_aug),' ',num2str(sample_idx),'/',num2str(numOrigSample)])
    temp_data = squeeze(data_orig(:,:,:,sample_idx));    
    
    temp_aug1 = temp_data(1:imSize_aug,1:imSize_aug); % up-left corner
    temp_aug2 = temp_data(1:imSize_aug,end-imSize_aug+1:end); % up-right corner 
    temp_aug3 = temp_data(end-imSize_aug+1:end,1:imSize_aug); % down-left corner
    temp_aug4 = temp_data(end-imSize_aug+1:end,end-imSize_aug+1:end); % down-right corner
    
    temp_flip = flip(temp_data,2);    

    data_withAug(:,:,1,numCount) = imresize(temp_data,[imSize_aug imSize_aug]);
    data_withAug(:,:,1,numCount+1) = temp_aug1;
    data_withAug(:,:,1,numCount+2) = temp_aug2;
    data_withAug(:,:,1,numCount+3) = temp_aug3;    
    data_withAug(:,:,1,numCount+4) = temp_aug4;

    data_withAug(:,:,1,numCount+5) = imresize(temp_flip,[imSize_aug imSize_aug]);
    data_withAug(:,:,1,numCount+6) = flip(temp_aug1,2);
    data_withAug(:,:,1,numCount+7) = flip(temp_aug2,2);
    data_withAug(:,:,1,numCount+8) = flip(temp_aug3,2);
    data_withAug(:,:,1,numCount+9) = flip(temp_aug4,2);   
    
    temp_set = set_orig(1, sample_idx);   
    set_withAug(1,numCount:1:numCount+9) = temp_set;  
    
    clear temp_data temp_flip temp_aug1 temp_aug2 temp_aug3 temp_aug4 temp_set
    
    if isempty(label_orig) ~= 1
        temp_label = label_orig(1, sample_idx);   
        label_withAug(1,numCount:1:numCount+9) = temp_label;  
        clear temp_label
    end
    
    numCount = numCount + numTimesAug;
end

