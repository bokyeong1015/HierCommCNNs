function outputData = func_DecFusion_SimpMedian(inputData)
                                                                    
    input_esti_score = inputData.esti_score; % input_esti_score: numClass x numSample x numModel
    true_label = inputData.true_label; % 1 x numSample

    outputData.esti_score = median(input_esti_score, 3);
    
    [~, outputData.esti_label] = max(outputData.esti_score);    
    outputData.true_label = true_label;
    outputData.acc_unbal = 100 * sum(true_label == outputData.esti_label)/length(true_label);

end
