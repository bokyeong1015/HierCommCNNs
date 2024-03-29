clc; clear; close all

%% General Setting
addpath('pipeline_modules_functions');
dataType_str = 'valid'; % data type e.g., train, valid, test
dataNorm_str = 'cEnh'; % normalization type
                      % e.g., Raw, iNor (illumination normalization), cEnh (contrast enhancement)
dataType_numFlag = 2; % 1 for train, 2 for valid, 3 for test

loadAlignFolder = ['result_FullyAutoFaceAlign_',dataType_str]; % folder name containing final alignments
loadFileList = dir([loadAlignFolder,'\finalSelect_',dataType_str,'*.mat']);  

imSize_orig = 48; % size of final aligned face
saveFileName_orig = ['imdb',num2str(imSize_orig),'_orig_SFEW',dataType_str,'_',dataNorm_str];

imSize_aug = 42; % size of augmented patches, 10 patches for each face
saveFileName_aug = ['imdb',num2str(imSize_aug),'_augX10_SFEW',dataType_str,'_',dataNorm_str];
% augmented by 10 times, through using 5 crops of size 42x42 
% (1 from resizing an original 48x48 face and 4 from extracting its 4 corners) and their horizontal flopping

%% Formation of input data ('imdb') for MatConvNet toolbox: original 48x48 aligned faces
% images.data: a 4-D data matrix containing images, (width)x(height)x(1)x(numImages), single type
% images.labels: a label vector, (1)x(numImages), double type, e.g. 1 for 'Anger', 2 for 'Disgust', etc
% images.set: a vector indicating data type, (1)x(numImages), 1 for train, 2 for valid, 3 for test
% images.orig_title: optional, to save original image file name
% meta.sets: strings indicating data type, e.g., {'train', 'val', 'test'}
% meta.classes: strings containing class labels, e.g., {'Anger', 'Disgust', 'Fear', 'Happy','Sad', 'Surprise', 'Neutral'} ;
            
%%%%%%%%%%%%%% Collecting original 48x48 aligned faces
numSample = size(loadFileList,1);

images.data = zeros(imSize_orig, imSize_orig, 1, numSample, 'single');
images.labels = zeros(1, numSample); % for training and validation data, you should properly fill images.labels 
images.set = zeros(1, numSample);
images.orig_title = cell(numSample, 1);

meta.sets = {'train', 'val', 'test'} ; 
meta.classes = {'Anger', 'Disgust', 'Fear', 'Happy',...,
                'Sad', 'Surprise', 'Neutral'} ;

for sample_idx = 1:1:numSample
    loadAlignMatName = [loadAlignFolder,...,
                        '\finalSelect_',dataType_str,'_s',num2str(sample_idx)];     
    
    load(loadAlignMatName,'final_align_face','orig_title','class_num')    
    disp(['imdb48 ',dataNorm_str,' ',num2str(sample_idx),'/',num2str(numSample), ': ', orig_title])
    if sum(isnan(final_align_face(:))) ~= 0; keyboard; end   
    if min(final_align_face(:)) ~= 0; keyboard; end   
    if max(final_align_face(:)) ~= 1; keyboard; end   
    
    %%%%%%%% contrast enhancement
    temp_histEqImg = histeq(single(final_align_face)); clear final_align_face
    
    min_value = min(temp_histEqImg(:));
    max_value = max(temp_histEqImg(:));
    final_align_face = (temp_histEqImg - min_value)./((max_value - min_value)+eps);      
    %%%%%%%%    
    
    images.data(:,:,1,sample_idx) = single(final_align_face);    
    images.set(1,sample_idx) = dataType_numFlag;
    images.orig_title{sample_idx,1} = orig_title;
        
    images.labels(1,sample_idx) = class_num;
    
    clear final_align_face orig_title class_num
end

%%%%%%%%%%%%%% Save           
disp(['Save ', saveFileName_orig])
save(saveFileName_orig,'images','meta');
orig_images = images; clear images

%% Formation of input data ('imdb') for MatConvNet toolbox: augmented 42x42 patches
[images.data, images.labels, images.set] =...
    func_AugDataLabel_x10(orig_images.data, orig_images.labels,...,
                          orig_images.set, imSize_aug); 
                      
%%%%%%%%%%%%%% Save           
disp(['Save ', saveFileName_aug])
save(saveFileName_aug,'images','meta');
clear images meta
