function [outFace_align, outLandmark, outFace_reCrop, align_info] =...
            func_FaceAlign_RuleBasedRotateCrop(face_im, output, resize_w_h, face_pos_old, fig_flag_lmkDetect_1)

%% Calculate Roatated Angle
% resize_w_h = 42;

orig_landmark = output.pred;
pos_lefteye_end = orig_landmark(20,:);
pos_righteye_end = orig_landmark(29,:);

leng1 = pos_righteye_end(1)-pos_lefteye_end(1);
leng2 = sqrt(sum((pos_lefteye_end-pos_righteye_end).^2));
alpha = acosd(leng1/leng2);

if pos_lefteye_end(2)<pos_righteye_end(2)
    alpha = alpha;        
elseif pos_lefteye_end(2)>pos_righteye_end(2)
    alpha = -alpha;
end

%%%%
xx_min = min(orig_landmark(:,1));
xx_max = max(orig_landmark(:,1));
yy_min = min(orig_landmark(:,2));
yy_max = max(orig_landmark(:,2));

del_xx = abs((xx_max - xx_min))/10;
del_yy = abs((yy_max - yy_min))/10;

xx_min = round(xx_min - del_xx);
xx_max = round(xx_max + del_xx);
yy_min = round(yy_min - del_yy);
yy_max = round(yy_max + del_yy);

if xx_min < 1; xx_min = 1; end
if yy_min < 1; yy_min = 1; end
if xx_max > size(face_im,2); xx_max = size(face_im,2); end
if yy_max > size(face_im,1); yy_max = size(face_im,1); end    

outFace_reCrop_0 = face_im(yy_min:yy_max, xx_min:xx_max);
outFace_reCrop = imresize(outFace_reCrop_0, [resize_w_h resize_w_h]);

%%%%
align_info.facePos0_fdOnly = face_pos_old;
align_info.facePos1_reCrop = [xx_min, yy_min, xx_max-xx_min+1, yy_max-yy_min+1]; % [x y w h]
align_info.landmark1_orig = orig_landmark; % 

%%%%

%% Rotation of Original Image
%         rot_orig_img = imrotate(face_im, alpha, 'nearest');  
rot_orig_img = imrotate(face_im, alpha, 'bicubic');  

%%%%
align_info.rotAngle = alpha;
%%%%

%% Rotation of Landmark Points

orig_landmark(round(orig_landmark(:,1))<1,1) = 1;
orig_landmark(round(orig_landmark(:,2))<1,2) = 1;
orig_landmark(round(orig_landmark(:,1))>size(face_im,2),1) = size(face_im,2);
orig_landmark(round(orig_landmark(:,2))>size(face_im,1),2) = size(face_im,1);    

for idx = 1:1:size(orig_landmark,1)
    landmark_img = zeros(size(face_im,1),size(face_im,2));
    
    aa = round(orig_landmark(idx,2));
    bb = round(orig_landmark(idx,1));

    landmark_img(aa,bb) = 1; % orig pts
    
    if aa ~= 1; landmark_img(aa-1,bb) = 0.5; end
    if bb ~= 1; landmark_img(aa,bb-1) = 0.5; end
    if aa ~= size(face_im,1); landmark_img(aa+1,bb) = 0.5; end
    if bb ~= size(face_im,2); landmark_img(aa,bb+1) = 0.5; end
    if aa ~=1 && bb ~=1; landmark_img(aa-1,bb-1) = 0.5; end
    if aa ~= size(face_im,1) && bb ~= size(face_im,2); landmark_img(aa+1,bb+1) = 0.5; end
    if aa ~= 1 && bb ~= size(face_im,2); landmark_img(aa-1,bb+1) = 0.5; end
    if aa ~= size(face_im,1) && bb ~= 1; landmark_img(aa+1,bb-1) = 0.5; end


    rot_landmark_img = imrotate(landmark_img, alpha, 'nearest');        
    [temp_rot_landmark_row temp_rot_landmark_col] = find(rot_landmark_img ~= 0);

    rot_landmark(idx,2) = round(mean(temp_rot_landmark_row));
    rot_landmark(idx,1) = round(mean(temp_rot_landmark_col));

    clear landmark_img rot_landmark_img
    clear temp_rot_landmark_row temp_rot_landmark_col
end   

rot_lefteye_end = [rot_landmark(20,:)];
rot_righteye_end = [rot_landmark(29,:)];

%% New Face Crop
x_new_orig = rot_landmark(1,1); % x of (1st point in left eye-brow)
y_new_orig = min(rot_landmark(1:10,2)); % highist pt among eye-brow's y       
w_new_orig = abs(rot_landmark(10,1)-rot_landmark(1,1)); 
            % (x of (1st point in left eye-brow)) - (x of (last point in right eye-brow))        
h_new_orig = abs(max(rot_landmark(39:43,2))-y_new_orig);
            % (lowest pt among y of (bottom lips)) - (highist pt among eye-brow's y))  

x_del = mean([diff(rot_landmark(1:5,1)); diff(rot_landmark(6:10,1))]);
% mean of distance btw eyebrow landmark pts 
y_del = mean(rot_landmark([20:23,26:29],2))-mean(rot_landmark(1:10,2));
% (eye y-position mean) - (eyebrow y-position mean)  

x_new = round(x_new_orig - x_del);
y_new = round(y_new_orig - y_del);
w_new = round(w_new_orig + x_del*2);
h_new = round(h_new_orig + y_del*2);

if x_new<1; x_new = 1; end
if y_new<1; y_new = 1; end
if x_new+w_new-1>size(rot_orig_img,2); w_new = size(rot_orig_img,2)-x_new+1; end
if y_new+h_new-1>size(rot_orig_img,1); h_new = size(rot_orig_img,1)-y_new+1; end   
 
face_rot_crop = rot_orig_img(y_new:y_new+h_new-1,x_new:x_new+w_new-1);
face_rot_crop_reSize = imresize(face_rot_crop, [resize_w_h resize_w_h]);    

%%%%
align_info.facePos2_afterRot = [x_new, y_new, w_new, h_new];
align_info.landmark2_afterRot = rot_landmark; % 
%%%%

%% Shift and Resize Landmark According to New Cropped Face

shift_rot_landmark = [rot_landmark(:,1)-x_new+1 rot_landmark(:,2)-y_new+1];
shift_reSize_rot_landmark(:,1) = round(shift_rot_landmark(:,1).*(resize_w_h/size(face_rot_crop,2)));    
shift_reSize_rot_landmark(:,2) = round(shift_rot_landmark(:,2).*(resize_w_h/size(face_rot_crop,1)));    

rot_reSize_lefteye_end = shift_reSize_rot_landmark(20,:);
rot_reSize_righteye_end = shift_reSize_rot_landmark(29,:);     

%%%%
align_info.landmark3_finalAlign = shift_reSize_rot_landmark; % 
%%%%

%% Figure
if fig_flag_lmkDetect_1 == 1
    x_old = face_pos_old(1);
    y_old = face_pos_old(2);
    w_old = face_pos_old(3);
    h_old = face_pos_old(4);

    figure;
    set(gcf,'position',[10 50 800 950]);

    subplot(3,2,1); 
    imagesc(face_im); colormap(gray);
    title(['Original Image']);   

    subplot(3,2,2); 
    imagesc(face_im); colormap(gray);
    title(['Prev Face Detection & Landmark Conf. = ',num2str(output.conf)]);      
    hold on; 
    plot([x_old x_old+w_old-1],[y_old y_old],'linewidth',2,'color','b');
    plot([x_old x_old+w_old-1],[y_old+h_old-1 y_old+h_old-1],'linewidth',2,'color','b');
    plot([x_old x_old],[y_old y_old+h_old-1],'linewidth',2,'color','b');
    plot([x_old+w_old-1 x_old+w_old-1],[y_old y_old+h_old-1],'linewidth',2,'color','b');

    plot(orig_landmark(:,1), orig_landmark(:,2),'r*','markersize',2);   
    plot([pos_lefteye_end(1) pos_righteye_end(1)],...,
        [pos_lefteye_end(2) pos_righteye_end(2)],':b','linewidth',2);         

    %%%%
    subplot(3,2,3); 
    imagesc(face_im); colormap(gray);
    title('ReCropping');   

    hold on; 
    plot([xx_min xx_max],[yy_min yy_min],'linewidth',2,'color','g');
    plot([xx_min xx_max],[yy_max yy_max],'linewidth',2,'color','g');
    plot([xx_min xx_min],[yy_min yy_max],'linewidth',2,'color','g');
    plot([xx_max xx_max],[yy_min yy_max],'linewidth',2,'color','g');

    plot(orig_landmark(:,1), orig_landmark(:,2),'r*','markersize',2);   
    plot([pos_lefteye_end(1) pos_righteye_end(1)],...,
        [pos_lefteye_end(2) pos_righteye_end(2)],':b','linewidth',2);       

    subplot(3,2,4); 
    imagesc(outFace_reCrop); colormap(gray);
    title('ReCropped Face');   

    %%%%
    subplot(3,2,5); 
    imagesc(rot_orig_img); colormap(gray);
    title(['Rotated Image: ',num2str(alpha),' deg']);       
    hold on; 
    plot(rot_landmark(:,1), rot_landmark(:,2),'r*','markersize',2);    
    plot([rot_lefteye_end(1) rot_righteye_end(1)],...,
        [rot_lefteye_end(2) rot_righteye_end(2)],':b','linewidth',2);    

    plot([x_new x_new+w_new-1],[y_new y_new],'linewidth',2,'color','c');
    plot([x_new x_new+w_new-1],[y_new+h_new-1 y_new+h_new-1],'linewidth',2,'color','c');
    plot([x_new x_new],[y_new y_new+h_new-1],'linewidth',2,'color','c');
    plot([x_new+w_new-1 x_new+w_new-1],[y_new y_new+h_new-1],'linewidth',2,'color','c');        

    subplot(3,2,6); 
    imagesc(face_rot_crop_reSize); colormap(gray);
    title('Final Rotated, Cropped & Resized Face');   
    hold on; 
    plot(shift_reSize_rot_landmark(:,1), shift_reSize_rot_landmark(:,2),'r*','markersize',2);  
    plot([rot_reSize_lefteye_end(1) rot_reSize_righteye_end(1)],...,
        [rot_reSize_lefteye_end(2) rot_reSize_righteye_end(2)],':b','linewidth',2);       
end

%%
outFace_align = face_rot_crop_reSize;
outLandmark = shift_reSize_rot_landmark;
end
