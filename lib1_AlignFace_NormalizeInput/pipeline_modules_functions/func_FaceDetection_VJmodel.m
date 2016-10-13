function [Face_location, Face_image, FaceCandi_set, end_flag] =...
            func_FaceDetection_VJmodel(Original_image, module_VJmodel_path, fig_flag_faceDetect)

%% Setting for image resizing-scales
% Img_Scalar = [100, 240, 320, 400, 512, 800, 1024, 150, 572];
% Img_Scalar = [32, 64, 128, 256, 512, 1024];
Img_Scalar = [16, 32, 64, 100, 128, 150, 240, 256, 320, 400, 512, 572, 800];

%% Setting for face detector filters
Img_Face_Detector = cell(4, 1);
Img_Face_Detector{1,1} = [module_VJmodel_path,'/haarcascade_frontalface_alt.xml'];
Img_Face_Detector{2,1} = [module_VJmodel_path,'/haarcascade_frontalface_alt_tree.xml'];
Img_Face_Detector{3,1} = [module_VJmodel_path,'/haarcascade_frontalface_alt2.xml'];
Img_Face_Detector{4,1} = [module_VJmodel_path,'/haarcascade_frontalface_default.xml'];

option.min_neighbors = 3;
%     .min_neighbors - OpenCV face detector parameter (>0). The lower the more 
%     likely to find a face as well as false positives.

%% Face Detection
Original_img_size = size(Original_image);
count = 1;
end_flag = -1;

Face_candidate_set = [];

for i = 1:length(Img_Scalar)
    if(Original_img_size(1,1) > Original_img_size(1,2)) % chaning the size of original image
        Target_image = imresize(Original_image, Img_Scalar(i)/Original_img_size(1,2));  
    else
        Target_image = imresize(Original_image, Img_Scalar(i)/Original_img_size(1,1)); 
    end
    Target_image_size = size(Target_image);        
    temp_img_scalar = ceil(Img_Scalar(i)/30);
    
    for j = 1:length(Img_Face_Detector)

        fd_h = cv.CascadeClassifier(Img_Face_Detector{j,1});        
        Face_cell = fd_h.detect(Target_image,'MinNeighbors', option.min_neighbors,...
                     'ScaleFactor', 1.2, 'MinSize',[temp_img_scalar temp_img_scalar]);

        if(isempty(Face_cell) ~= 1) %% When Finding Face

            for temp_idx = 1:1:length(Face_cell)
                Face(temp_idx,:) = Face_cell{temp_idx};
            end

            end_flag = 1;
            Face_candidate_set = [Face_candidate_set; 
                repmat(i,size(Face,1),1), repmat(j,size(Face,1),1),...,
                Face(:,1)/Target_image_size(1,1),...,
                Face(:,2)/Target_image_size(1,2), Face(:,3)/Target_image_size(1,1)];
            
            count = count+size(Face,1);
            clear Face_cell Face
        end           
    end    
end

%% Face Selection
if  size(Face_candidate_set,1) ~= 0
    flag_x_leng = ((uint32(Original_img_size(1,1)*Face_candidate_set(:,3)))<size(Original_image,1));
    flag_y_leng = ((uint32(Original_img_size(1,2)*Face_candidate_set(:,4)))<size(Original_image,2));
    flag_w_leng = ((uint32(Original_img_size(1,1)*Face_candidate_set(:,5)))<size(Original_image,1));
    right_idx = find(flag_x_leng.*flag_y_leng.*flag_w_leng == 1);
    
    Face_candidate_set = Face_candidate_set(right_idx, :);
end

if  size(Face_candidate_set,1) == 0
    end_flag = -1;
elseif  size(Face_candidate_set,1) == 1
    Face_size = [uint32(Original_img_size(1,1) * Face_candidate_set(1,3)),...,
    uint32(Original_img_size(1,2) * Face_candidate_set(1,4)),...,
    uint32(Original_img_size(1,1) * Face_candidate_set(1,5))];
    x_final = Face_size(1);
    y_final = Face_size(2);
    w_final = Face_size(3);
    h_final = Face_size(3);        
    fprintf('   only 1 candidate exists !!! ');
else
    for candi_idx = 1:1:size(Face_candidate_set,1)
        Face_size = [uint32(Original_img_size(1,1) * Face_candidate_set(candi_idx,3)),...,
            uint32(Original_img_size(1,2) * Face_candidate_set(candi_idx,4)),...,
            uint32(Original_img_size(1,1) * Face_candidate_set(candi_idx,5))];
        F_zero = find(Face_size == 0);
        Face_size(F_zero) = 1;

        x = Face_size(1);
        y = Face_size(2);
        w = Face_size(3);
        h = Face_size(3);    

        faceSize_collec(candi_idx,:) = [x y w h];
        clear x y w h
    end

    min_faceSize_collec = min(faceSize_collec);
    max_faceSize_collec = max(faceSize_collec);
    
    for pos_idx = 1:1:size(faceSize_collec,2)
        temp_data = faceSize_collec(:,pos_idx);
        range_arr = [0 0.001 0.005 0.05 0.25 0.50 0.75 0.95 0.995 0.999 1];
        temp_range = quantile(double(temp_data), range_arr);      

        faceSize_final(1,pos_idx) = uint32(temp_range(range_arr == 0.5));
    end
    
    x_final = faceSize_final(1);
    y_final = faceSize_final(2);
    w_final = faceSize_final(3);
    h_final = faceSize_final(4);    

    if (x_final+w_final-1 > size(Original_image,2)) || (y_final+w_final-1 > size(Original_image,1));
        w_final = w_final-1;
    end
end


if(end_flag ~= -1)
    Face_location = double([x_final y_final w_final h_final]);
    F_zero = find(Face_location == 0);
    Face_location(F_zero) = 1;    
    Face_image = Original_image(y_final:(y_final+w_final-1), x_final:(x_final+w_final-1),:);
    
    for candi_idx = 1:1:size(Face_candidate_set,1)
        Face_size = [uint32(Original_img_size(1,1) * Face_candidate_set(candi_idx,3)),...,
            uint32(Original_img_size(1,2) * Face_candidate_set(candi_idx,4)),...,
            uint32(Original_img_size(1,1) * Face_candidate_set(candi_idx,5))];
        F_zero = find(Face_size == 0);
        Face_size(F_zero) = 1;

        x = Face_size(1);
        y = Face_size(2);
        w = Face_size(3);
        h = Face_size(3);    

        FaceCandi_set(candi_idx,:) = [x y w h];
        clear x y w h
    end
else    
    Face_location = [0 0 0 0];
    Face_image = -1;
    FaceCandi_set = [];
    fprintf('   can NOT detect face !!! ');
end

%% Figure
if fig_flag_faceDetect == 1
    figure;
    
    subplot(1,2,1)
    imagesc(Original_image); colormap(gray) 
    hold on
    
    if(end_flag ~= -1)
        for candi_idx = 1:1:size(Face_candidate_set,1)
                Face_size = [uint32(Original_img_size(1,1) * Face_candidate_set(candi_idx,3)),...,
                    uint32(Original_img_size(1,2) * Face_candidate_set(candi_idx,4)),...,
                    uint32(Original_img_size(1,1) * Face_candidate_set(candi_idx,5))];
                F_zero = find(Face_size == 0);
                Face_size(F_zero) = 1;

                x = Face_size(1);
                y = Face_size(2);
                w = Face_size(3);
                h = Face_size(3);    

                plot([x x+h-1],[y y],'linewidth',2,'color','r');
                plot([x x+h-1],[y+h-1 y+h-1],'linewidth',2,'color','r');
                plot([x x],[y y+h-1],'linewidth',2,'color','r');
                plot([x+h-1 x+h-1],[y y+h-1],'linewidth',2,'color','r');

                faceSize_collec(candi_idx,:) = [x y w h];
                clear x y w h
        end

        plot([x_final x_final+h_final-1],[y_final y_final],'-b','linewidth',3);
        plot([x_final x_final+h_final-1],[y_final+h_final-1 y_final+h_final-1],'-b','linewidth',3);
        plot([x_final x_final],[y_final y_final+h_final-1],'-b','linewidth',3);
        plot([x_final+h_final-1 x_final+h_final-1],[y_final y_final+h_final-1],'-b','linewidth',3);
   
    end
        
    subplot(1,2,2)    
    imagesc(Face_image);    

end

    
end


