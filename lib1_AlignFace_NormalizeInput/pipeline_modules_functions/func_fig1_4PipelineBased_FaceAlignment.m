% figure;
figure('name',[dataType_str,...,
              ', sample_idx = ', num2str(sample_idx),...,
              ' | ',orig_title]);

set(gcf,'position',position_fig1);

%%
if end_flag_lmk_1 == 1
    alpha_1 = align_info_1.rotAngle;
    rot_orig_img_1 = imrotate(raw_img_gray, alpha_1, 'bicubic');  
    rot_landmark_1 = align_info_1.landmark2_afterRot;
    x_align_1 = align_info_1.facePos2_afterRot(1);
    y_align_1 = align_info_1.facePos2_afterRot(2);
    w_align_1 = align_info_1.facePos2_afterRot(3);
    h_align_1 = align_info_1.facePos2_afterRot(4);    
end

if end_flag_lmk_2 == 1
    alpha_2 = align_info_2.rotAngle;
    rot_orig_img_2 = imrotate(raw_img_histeq, alpha_2, 'bicubic');  
    rot_landmark_2 = align_info_2.landmark2_afterRot;
    x_align_2 = align_info_2.facePos2_afterRot(1);
    y_align_2 = align_info_2.facePos2_afterRot(2);
    w_align_2 = align_info_2.facePos2_afterRot(3);
    h_align_2 = align_info_2.facePos2_afterRot(4);       
end

if end_flag_lmk_3 == 1
    alpha_3 = align_info_3.rotAngle;
    rot_orig_img_3 = imrotate(raw_img_gray, alpha_3, 'bicubic');  
    rot_landmark_3 = align_info_3.landmark2_afterRot;
    x_align_3 = align_info_3.facePos2_afterRot(1);
    y_align_3 = align_info_3.facePos2_afterRot(2);
    w_align_3 = align_info_3.facePos2_afterRot(3);
    h_align_3 = align_info_3.facePos2_afterRot(4);    
end

if end_flag_lmk_4 == 1
    alpha_4 = align_info_4.rotAngle;
    rot_orig_img_4 = imrotate(raw_img_histeq, alpha_4, 'bicubic');  
    rot_landmark_4 = align_info_4.landmark2_afterRot;
    x_align_4 = align_info_4.facePos2_afterRot(1);
    y_align_4 = align_info_4.facePos2_afterRot(2);
    w_align_4 = align_info_4.facePos2_afterRot(3);
    h_align_4 = align_info_4.facePos2_afterRot(4);       
end
            
%% 1. 'VjFd-Raw'
subplot(4, 4, 1);
imagesc(raw_img_color);
set(gca,'xtick',[],'ytick',[]);
hold on

if end_flag_fd_1 == 1
    for candi_idx = 1:1:size(faceCandi_set_1,1)
        x = faceCandi_set_1(candi_idx,1); y = faceCandi_set_1(candi_idx,2);
        w = faceCandi_set_1(candi_idx,3); h = faceCandi_set_1(candi_idx,4);
        
        plot([x x+w-1],[y y],'linewidth',2,'color',[0.95 0.3 0.1]);
        plot([x x+w-1],[y+h-1 y+h-1],'linewidth',2,'color',[0.95 0.3 0.1]);
        plot([x x],[y y+h-1],'linewidth',2,'color',[0.95 0.3 0.1]);
        plot([x+w-1 x+w-1],[y y+h-1],'linewidth',2,'color',[0.95 0.3 0.1]);
        clear x y w h
    end
    
    x_final = face_loc_1(1); y_final = face_loc_1(2);
    w_final = face_loc_1(3); h_final = face_loc_1(4);
    
    plot([x_final x_final+w_final-1],[y_final y_final],'-b','linewidth',3);
    plot([x_final x_final+w_final-1],[y_final+h_final-1 y_final+h_final-1],'-b','linewidth',3);
    plot([x_final x_final],[y_final y_final+h_final-1],'-b','linewidth',3);
    plot([x_final+w_final-1 x_final+w_final-1],[y_final y_final+h_final-1],'-b','linewidth',3);
    % pause(0.5)    
end
title('Face Detection','fontsize',12);      

subplot(4, 4, 2);
imagesc(face_im_1); 
set(gca,'xtick',[],'ytick',[]);
title('Face Detection Result','fontsize',12); 

subplot(4, 4, 3);
if end_flag_lmk_1 == 1
    imagesc(rot_orig_img_1); colormap(gray); hold on; 
    set(gca,'xtick',[],'ytick',[]);
    plot(rot_landmark_1(:,1), rot_landmark_1(:,2),'r*','markersize',2);    
    plot([rot_landmark_1(20,1) rot_landmark_1(29,1)],...,
         [rot_landmark_1(20,2) rot_landmark_1(29,2)],'-g','linewidth',2);    
    plot([x_align_1 x_align_1+w_align_1-1],[y_align_1 y_align_1],'linewidth',2,'color','c');
    plot([x_align_1 x_align_1+w_align_1-1],[y_align_1+h_align_1-1 y_align_1+h_align_1-1],'linewidth',2,'color','c');
    plot([x_align_1 x_align_1],[y_align_1 y_align_1+h_align_1-1],'linewidth',2,'color','c');
    plot([x_align_1+w_align_1-1 x_align_1+w_align_1-1],[y_align_1 y_align_1+h_align_1-1],'linewidth',2,'color','c');    
else
    imagesc(raw_img_gray); colormap(gray); hold on; 
    set(gca,'xtick',[],'ytick',[]);    
end
title('LMK Detection & Align','fontsize',12);    

subplot(4, 4, 4);
if end_flag_lmk_1 == 1
    imagesc(outFace_align_1); colormap(gray); hold on; 
    set(gca,'xtick',[],'ytick',[]);
    plot(outLandmark_1(:,1), outLandmark_1(:,2),'r*','markersize',4);  
    plot([outLandmark_1(20,1) outLandmark_1(29,1)],...,
         [outLandmark_1(20,2) outLandmark_1(29,2)],'-g','linewidth',2);    
else
    imagesc(ones(48,48).*(-1)); colormap(gray); hold on; 
    set(gca,'xtick',[],'ytick',[]);
end
title(['48x48 Face, Conf. = ', num2str(lmk_conf_1,2)],'fontsize',12,'fontweight','bold');    

%% 2. 'VjFd-HistEq'
subplot(4, 4, 1+4);
imagesc(raw_img_histeq);
set(gca,'xtick',[],'ytick',[]);
hold on

if end_flag_fd_2 == 1
    for candi_idx = 1:1:size(faceCandi_set_2,1)
        x = faceCandi_set_2(candi_idx,1); y = faceCandi_set_2(candi_idx,2);
        w = faceCandi_set_2(candi_idx,3); h = faceCandi_set_2(candi_idx,4);
        
        plot([x x+w-1],[y y],'linewidth',2,'color',[0.95 0.3 0.1]);
        plot([x x+w-1],[y+h-1 y+h-1],'linewidth',2,'color',[0.95 0.3 0.1]);
        plot([x x],[y y+h-1],'linewidth',2,'color',[0.95 0.3 0.1]);
        plot([x+w-1 x+w-1],[y y+h-1],'linewidth',2,'color',[0.95 0.3 0.1]);
        clear x y w h
    end
    
    x_final = face_loc_2(1); y_final = face_loc_2(2);
    w_final = face_loc_2(3); h_final = face_loc_2(4);
    
    plot([x_final x_final+w_final-1],[y_final y_final],'-b','linewidth',3);
    plot([x_final x_final+w_final-1],[y_final+h_final-1 y_final+h_final-1],'-b','linewidth',3);
    plot([x_final x_final],[y_final y_final+h_final-1],'-b','linewidth',3);
    plot([x_final+w_final-1 x_final+w_final-1],[y_final y_final+h_final-1],'-b','linewidth',3);
    % pause(0.5)    
end
title('Face Detection','fontsize',12);      

subplot(4, 4, 2+4);
imagesc(face_im_2); 
set(gca,'xtick',[],'ytick',[]);
title('Face Detection Result','fontsize',12); 

subplot(4, 4, 3+4);
if end_flag_lmk_2 == 1
    imagesc(rot_orig_img_2); colormap(gray); hold on; 
    set(gca,'xtick',[],'ytick',[]);
    plot(rot_landmark_2(:,1), rot_landmark_2(:,2),'r*','markersize',2);    
    plot([rot_landmark_2(20,1) rot_landmark_2(29,1)],...,
         [rot_landmark_2(20,2) rot_landmark_2(29,2)],'-g','linewidth',2);    
    plot([x_align_2 x_align_2+w_align_2-1],[y_align_2 y_align_2],'linewidth',2,'color','c');
    plot([x_align_2 x_align_2+w_align_2-1],[y_align_2+h_align_2-1 y_align_2+h_align_2-1],'linewidth',2,'color','c');
    plot([x_align_2 x_align_2],[y_align_2 y_align_2+h_align_2-1],'linewidth',2,'color','c');
    plot([x_align_2+w_align_2-1 x_align_2+w_align_2-1],[y_align_2 y_align_2+h_align_2-1],'linewidth',2,'color','c');    
else
    imagesc(raw_img_histeq); colormap(gray); hold on; 
    set(gca,'xtick',[],'ytick',[]);    
end
title('LMK Detection & Align','fontsize',12);    

subplot(4, 4, 4+4);
if end_flag_lmk_2 == 1
    imagesc(outFace_align_2); colormap(gray); hold on; 
    set(gca,'xtick',[],'ytick',[]);
    plot(outLandmark_2(:,1), outLandmark_2(:,2),'r*','markersize',4);  
    plot([outLandmark_2(20,1) outLandmark_2(29,1)],...,
         [outLandmark_2(20,2) outLandmark_2(29,2)],'-g','linewidth',2);    
else
    imagesc(ones(48,48).*(-1)); colormap(gray); hold on; 
    set(gca,'xtick',[],'ytick',[]);
end
title(['48x48 Face, Conf. = ', num2str(lmk_conf_2,2)],'fontsize',12,'fontweight','bold');    

%% 3. 'XZhuFd-Raw'
subplot(4, 4, 1+8);
imagesc(raw_img_color);
set(gca,'xtick',[],'ytick',[]);
hold on

if end_flag_fd_3 == 1
    func_ZRmodel_showboxes_modi(raw_img_color, bs_3, posemap)
    
    x_final = face_loc_3(1); y_final = face_loc_3(2);
    w_final = face_loc_3(3); h_final = face_loc_3(4);
    
    plot([x_final x_final+w_final-1],[y_final y_final],'-b','linewidth',3);
    plot([x_final x_final+w_final-1],[y_final+h_final-1 y_final+h_final-1],'-b','linewidth',3);
    plot([x_final x_final],[y_final y_final+h_final-1],'-b','linewidth',3);
    plot([x_final+w_final-1 x_final+w_final-1],[y_final y_final+h_final-1],'-b','linewidth',3);
    % pause(0.5)    
end
title('Face Detection','fontsize',12);      

subplot(4, 4, 2+8);
imagesc(face_im_3); 
set(gca,'xtick',[],'ytick',[]);
title('Face Detection Result','fontsize',12); 

subplot(4, 4, 3+8);
if end_flag_lmk_3 == 1
    imagesc(rot_orig_img_3); colormap(gray); hold on; 
    set(gca,'xtick',[],'ytick',[]);
    plot(rot_landmark_3(:,1), rot_landmark_3(:,2),'r*','markersize',2);    
    plot([rot_landmark_3(20,1) rot_landmark_3(29,1)],...,
         [rot_landmark_3(20,2) rot_landmark_3(29,2)],'-g','linewidth',2);    
    plot([x_align_3 x_align_3+w_align_3-1],[y_align_3 y_align_3],'linewidth',2,'color','c');
    plot([x_align_3 x_align_3+w_align_3-1],[y_align_3+h_align_3-1 y_align_3+h_align_3-1],'linewidth',2,'color','c');
    plot([x_align_3 x_align_3],[y_align_3 y_align_3+h_align_3-1],'linewidth',2,'color','c');
    plot([x_align_3+w_align_3-1 x_align_3+w_align_3-1],[y_align_3 y_align_3+h_align_3-1],'linewidth',2,'color','c');    
else
    imagesc(raw_img_gray); colormap(gray); hold on; 
    set(gca,'xtick',[],'ytick',[]);    
end
title('LMK Detection & Align','fontsize',12);    

subplot(4, 4, 4+8);
if end_flag_lmk_3 == 1
    imagesc(outFace_align_3); colormap(gray); hold on; 
    set(gca,'xtick',[],'ytick',[]);
    plot(outLandmark_3(:,1), outLandmark_3(:,2),'r*','markersize',4);  
    plot([outLandmark_3(20,1) outLandmark_3(29,1)],...,
         [outLandmark_3(20,2) outLandmark_3(29,2)],'-g','linewidth',2);    
else
    imagesc(ones(48,48).*(-1)); colormap(gray); hold on; 
    set(gca,'xtick',[],'ytick',[]);
end
title(['48x48 Face, Conf. = ', num2str(lmk_conf_3,2)],'fontsize',12,'fontweight','bold');    


%% 4. 'XZhuFd-HistEq'
subplot(4, 4, 1+12);
imagesc(raw_img_color);
set(gca,'xtick',[],'ytick',[]);
hold on

if end_flag_fd_4 == 1
    func_ZRmodel_showboxes_modi(raw_img_color, bs_4, posemap)
    
    x_final = face_loc_4(1); y_final = face_loc_4(2);
    w_final = face_loc_4(3); h_final = face_loc_4(4);
    
    plot([x_final x_final+w_final-1],[y_final y_final],'-b','linewidth',3);
    plot([x_final x_final+w_final-1],[y_final+h_final-1 y_final+h_final-1],'-b','linewidth',3);
    plot([x_final x_final],[y_final y_final+h_final-1],'-b','linewidth',3);
    plot([x_final+w_final-1 x_final+w_final-1],[y_final y_final+h_final-1],'-b','linewidth',3);
    % pause(0.5)    
end
title('Face Detection','fontsize',12);      

subplot(4, 4, 2+12);
imagesc(face_im_4); 
set(gca,'xtick',[],'ytick',[]);
title('Face Detection Result','fontsize',12); 

subplot(4, 4, 3+12);
if end_flag_lmk_4 == 1
    imagesc(rot_orig_img_4); colormap(gray); hold on; 
    set(gca,'xtick',[],'ytick',[]);
    plot(rot_landmark_4(:,1), rot_landmark_4(:,2),'r*','markersize',2);    
    plot([rot_landmark_4(20,1) rot_landmark_4(29,1)],...,
         [rot_landmark_4(20,2) rot_landmark_4(29,2)],'-g','linewidth',2);    
    plot([x_align_4 x_align_4+w_align_4-1],[y_align_4 y_align_4],'linewidth',2,'color','c');
    plot([x_align_4 x_align_4+w_align_4-1],[y_align_4+h_align_4-1 y_align_4+h_align_4-1],'linewidth',2,'color','c');
    plot([x_align_4 x_align_4],[y_align_4 y_align_4+h_align_4-1],'linewidth',2,'color','c');
    plot([x_align_4+w_align_4-1 x_align_4+w_align_4-1],[y_align_4 y_align_4+h_align_4-1],'linewidth',2,'color','c');    
else
    imagesc(raw_img_histeq); colormap(gray); hold on; 
    set(gca,'xtick',[],'ytick',[]);    
end
title('LMK Detection & Align','fontsize',12);    

subplot(4, 4, 4+12);
if end_flag_lmk_4 == 1
    imagesc(outFace_align_4); colormap(gray); hold on; 
    set(gca,'xtick',[],'ytick',[]);
    plot(outLandmark_4(:,1), outLandmark_4(:,2),'r*','markersize',4);  
    plot([outLandmark_4(20,1) outLandmark_4(29,1)],...,
         [outLandmark_4(20,2) outLandmark_4(29,2)],'-g','linewidth',2);    
else
    imagesc(ones(48,48).*(-1)); colormap(gray); hold on; 
    set(gca,'xtick',[],'ytick',[]);
end
title(['48x48 Face, Conf. = ', num2str(lmk_conf_4,2)],'fontsize',12,'fontweight','bold');    

%%
clear alpha_1 rot_orig_img_1 rot_landmark_1 
clear x_align_1 y_align_1 w_align_1 h_align_1
clear alpha_2 rot_orig_img_2 rot_landmark_2 
clear x_align_2 y_align_2 w_align_2 h_align_2    
clear alpha_3 rot_orig_img_3 rot_landmark_3 
clear x_align_3 y_align_3 w_align_3 h_align_3
clear alpha_4 rot_orig_img_4 rot_landmark_4 
clear x_align_4 y_align_4 w_align_4 h_align_4