xx = final_align_info.facePos0_fdOnly(1);
yy = final_align_info.facePos0_fdOnly(2);
ww = final_align_info.facePos0_fdOnly(3);
hh = final_align_info.facePos0_fdOnly(4);

figure;
set(gcf, 'position', position_fig2);

subplot(1,2,1)
imagesc(raw_img_color); hold on;
plot([xx xx+ww-1],[yy yy],'-b','linewidth',3);
plot([xx xx+ww-1],[yy+hh-1 yy+hh-1],'-b','linewidth',3);
plot([xx xx],[yy yy+hh-1],'-b','linewidth',3);
plot([xx+ww-1 xx+ww-1],[yy yy+hh-1],'-b','linewidth',3);   
title(['Raw Img, s = ',num2str(sample_idx)],'fontweight','bold')

subplot(1,2,2)
imagesc(final_align_face); 
colormap(gray); colorbar
title(['Final Img, Case ',case_flag_str,': pipeline ',num2str(select_pipeline_idx)],'fontweight','bold')    

 clear xx yy ww hh