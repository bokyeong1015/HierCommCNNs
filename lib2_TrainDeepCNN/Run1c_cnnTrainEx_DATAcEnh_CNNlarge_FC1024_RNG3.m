clc; clear; close all

%%% This example code gives CNN configurations and other learning parameter settings.

%%% However, for a complete code used in our article, you need to
%%% have two external DBs (FER-2013 and TFD) and 
%%% preprocess these DBs using functions in lib1_AlignFace_NormalizeInput.
%%% -- FER-2013: https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data
%%% -- Toronto Face Database (TFD): personal contact to the author of TFD technical report
%%% In addition, you need to modify the code as follows.
%%%   1) Pretrain a deep CNN using the FER-2013 database and the TFD
%%%   2) Set the pretrained deep CNN from '1)' as initialization,
%%%      and finetune this deep CNN using the SFEW2.0 data.
%%% Here, we do not provide any external DBs and pre-trained models due to
%%% the memory issue for uploading. Please carefully read our article and modify this code properly. 

%% General Setting
addpath MatConvNet_v1b8
% For training deep CNNs, we used the MatConvNet toolbox
% (version1.0-beta8) on NVIDIA GeForce GTX 690 GPUs
% For detailed information, please visit http://www.vlfeat.org/matconvnet/

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rngNum = 3; % rng number for random weight initialization, e.g., 1,2,3
num_fcHiddenNeuron = 1024; % # neurons in the fully-connected hidden layer, e.g., 3072, 2048, 1024, 512
prob_fcDropout = 0.5; % dropout probability in the fully-connected hidden layer, e.g., 0.8, 0.5

flag_cnn = 'lar'; % size of CNN receptive fields, e.g., 'med' for medium, 'lar' for large, 'sma' for small.
% You should properly set the below CNN setting corresponding to flag_cnn

flag_input = 'cEnh'; 
% type of input normalization, e.g., 'raw' for raw images, 'iNor' for illumination normalization, 
% and 'cEnh' for contrast enhancement.
% You should load the proper input data (imdb) corresponding to flag_input

save_folderName = ['DATA',flag_input,'_CNN',flag_cnn,...,
                    '_FC',num2str(num_fcHiddenNeuron),'_RNG',num2str(rngNum)];
% folder in which the resulting deep CNNs are saved 
opts.expDir = fullfile('data',save_folderName) ;
if ~exist(opts.expDir, 'dir'), mkdir(opts.expDir) ; end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% input data for training deep CNNs
% concatenate training data and validation data
imdb1 = load(['imdb42_augX10_SFEWtrain_cEnh']) ; 
imdb2 = load(['imdb42_augX10_SFEWvalid_cEnh']) ; 

imdb.images.data = cat(4, imdb1.images.data, imdb2.images.data);
imdb.images.labels = cat(2, imdb1.images.labels, imdb2.images.labels);
imdb.images.set = cat(2, imdb1.images.set, imdb2.images.set);
imdb.meta = imdb1.meta;

clear imdb1 imdb2
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
opts.useGpu = false;
% If you want to use gpu setup, you may need to re-compile matconvnet-1.0-beta8.tar.gz.
% After re-compiling, files of {vl_nnconv.mexa64, vl_nnnormalize.mexa64, vl_nnpool.mexa64}
% could be working on your gpu device properly.
% For detailed information, please visit http://www.vlfeat.org/matconvnet/

if opts.useGpu == 1
    gpuDeviceNum = input('gpuDeviceNum: ')
    gpuDevice(gpuDeviceNum)
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% setting for stochastic gradient descent in training deep CNNs
opts.batchSize = 200 ;
opts.numEpochs = 100 ;
opts.continue = true ;

% opts.learningRate = [0.01*ones(1,25), 0.005*ones(1,25), 0.0025*ones(1,25), 0.00125*ones(1,25)]; % for pre-training
opts.learningRate = [0.004*ones(1,25), 0.002*ones(1,25), 0.001*ones(1,25), 0.0005*ones(1,25)]; % for fine-tuning
  
       
%% Training Deep CNNs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CNN configuration
net.layers = {} ;

%%% Conv1 - MaxPool1
rng(rngNum)
net.layers{end+1} = struct('type', 'conv', ...
                       'filters', 0.01 * randn(7, 7, 1, 32, 'single'), ...
                       'biases', 0.1 * ones(1, 32, 'single'), ...
                       'stride', 1, ...
                       'pad', 3, ...
                       'filtersLearningRate', 1, ...
                       'biasesLearningRate', 1, ...
                       'filtersWeightDecay', 1/5, ...
                       'biasesWeightDecay', 0) ;
net.layers{end+1} = struct('type', 'relu') ;
net.layers{end+1} = struct('type', 'pool', ...
                       'method', 'max', ...
                       'pool', [2 2], ...
                       'stride', 2, ...
                       'pad', 0) ;
                   
%%% Conv2 - MaxPool2
rng(rngNum)
net.layers{end+1} = struct('type', 'conv', ...
                       'filters', 0.01 * randn(7, 7, 32, 32, 'single'), ...
                       'biases', 0.1 * ones(1, 32, 'single'), ...
                       'stride', 1, ...
                       'pad', 2, ...
                       'filtersLearningRate', 1, ...
                       'biasesLearningRate', 1, ...
                       'filtersWeightDecay', 1/5, ...
                       'biasesWeightDecay', 0) ;
net.layers{end+1} = struct('type', 'relu') ;
net.layers{end+1} = struct('type', 'pool', ...
                       'method', 'max', ...
                       'pool', [2 2], ...
                       'stride', 2, ...
                       'pad', [1, 0, 1, 0]) ;

%%% Conv3 - MaxPool3
rng(rngNum)
net.layers{end+1} = struct('type', 'conv', ...
                       'filters', 0.01 * randn(7, 7, 32, 64,'single'), ...
                       'biases', 0.1 * ones(1, 64,'single'), ...
                       'stride', 1, ...
                       'pad', 3, ...
                       'filtersLearningRate', 1, ...
                       'biasesLearningRate', 1, ...
                       'filtersWeightDecay', 1/5, ...
                       'biasesWeightDecay', 0) ;
net.layers{end+1} = struct('type', 'relu') ;
net.layers{end+1} = struct('type', 'pool', ...
                       'method', 'max', ...
                       'pool', [2 2], ...
                       'stride', 2, ...
                       'pad', 0) ;               
                   
%%% Fc Hidden
rng(rngNum)
net.layers{end+1} = struct('type', 'conv', ...
                       'filters', 0.001 * randn(5, 5, 64, num_fcHiddenNeuron,'single'),...
                       'biases', 0.01 * ones(1, num_fcHiddenNeuron,'single'), ...
                       'stride', 1, ...
                       'pad', 0, ...
                       'filtersLearningRate', 1, ...
                       'biasesLearningRate', 1, ...
                       'filtersWeightDecay', 1/5, ...
                       'biasesWeightDecay', 0) ;
net.layers{end+1} = struct('type', 'relu') ;
net.layers{end+1} = struct('type', 'dropout', ...
                           'rate', prob_fcDropout) ;

%%% Fc Output
rng(rngNum)
net.layers{end+1} = struct('type', 'conv', ...
                       'filters', zeros(1, 1, num_fcHiddenNeuron, 7,'single'), ...
                       'biases', zeros(1, 7, 'single'), ...
                       'stride', 1, ...
                       'pad', 0, ...
                       'filtersLearningRate', 1, ...
                       'biasesLearningRate', 1, ...
                       'filtersWeightDecay', 4, ...
                       'biasesWeightDecay', 0) ;
net.layers{end+1} = struct('type', 'softmaxloss') ;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% saving the initialized deep CNN
net_init = net;
save(fullfile(opts.expDir,'net_init.mat'),'net_init');
clear net_init


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% starting to train deep CNN !
tstart = tic;
[net, info] = cnn_train(net, imdb, @getBatch, ...
    opts, 'val', find(imdb.images.set == 2)) ;
time_cnnTrain = toc(tstart);

disp(' ')
disp(save_folderName);
disp(['time for training cnn = ',num2str(time_cnnTrain),' sec',...,
    ' = ',num2str(time_cnnTrain/3600),' hour'])

save(fullfile(opts.expDir,['opts_time','_',save_folderName,'.mat']),'opts','time_cnnTrain');
saveas(gcf,fullfile(opts.expDir,['net_train_cost','_',save_folderName,'.fig']),'fig');
