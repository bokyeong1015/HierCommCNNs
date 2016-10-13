function func_ZRmodel_showboxes_modi(im, boxes, posemap)

% showboxes(im, boxes)
% Draw boxes on top of image.

% imagesc(im);
% hold on;
% axis image;
% axis off;

count = 1;

for b = boxes,
    partsize = b.xy(1,3)-b.xy(1,1)+1;
    tx = (min(b.xy(:,1)) + max(b.xy(:,3)))/2;
    ty = min(b.xy(:,2)) - partsize/2;
    
    for i = size(b.xy,1):-1:1;
        x1 = b.xy(i,1);
        y1 = b.xy(i,2);
        x2 = b.xy(i,3);
        y2 = b.xy(i,4);
        line([x1 x1 x2 x2 x1]', [y1 y2 y2 y1 y1]', 'color', [0.6 0.6 0.6], 'linewidth', 1);
        
        plot((x1+x2)/2,(y1+y2)/2,'m.','markersize',6);
    end
    
    %%
    temp_bs = b.xy;

    bs_1_xy(:,1) = (temp_bs(:,1)+temp_bs(:,3))/2;
    bs_1_xy(:,2) = (temp_bs(:,2)+temp_bs(:,4))/2;
    %%%%
    xx_min = min(bs_1_xy(:,1));
    xx_max = max(bs_1_xy(:,1));
    yy_min = min(bs_1_xy(:,2));
    yy_max = max(bs_1_xy(:,2));

    del_xx = abs((xx_max - xx_min))/10;
    del_yy = abs((yy_max - yy_min))/10;

    xx_min = round(xx_min - del_xx);
    xx_max = round(xx_max + del_xx);
    yy_min = round(yy_min - del_yy);
    yy_max = round(yy_max + del_yy);

%     if xx_min < 1; xx_min = 1; end
%     if yy_min < 1; yy_min = 1; end
%     if xx_max > size(raw_img_gray,2); xx_max = size(raw_img_gray,2); end
%     if yy_max > size(raw_img_gray,1); yy_max = size(raw_img_gray,1); end    

    x = xx_min;
    y = yy_min;
    w = xx_max - xx_min + 1;
    h = yy_max - yy_min + 1;
    faces_position = [x, y, w, h];

    if count == 1        
        plot([x x+w-1],[y y],'-b','linewidth',3);
        plot([x x+w-1],[y+h-1 y+h-1],'-b','linewidth',3);
        plot([x x],[y y+h-1],'-b','linewidth',3);
        plot([x+w-1 x+w-1],[y y+h-1],'-b','linewidth',3);
    else
        plot([x x+w-1],[y y],'linewidth',2,'color',[0.95 0.3 0.1]);
        plot([x x+w-1],[y+h-1 y+h-1],'linewidth',2,'color',[0.95 0.3 0.1]);
        plot([x x],[y y+h-1],'linewidth',2,'color',[0.95 0.3 0.1]);
        plot([x+w-1 x+w-1],[y y+h-1],'linewidth',2,'color',[0.95 0.3 0.1]);      
    end
    
%     text(tx,ty, num2str(posemap(b.c)),'fontsize',16,'color',[0.2 0.8 0.4],'fontweight','bold');

    count = count+1;
    
    clear bs_1_xy temp_bs
            
end
drawnow;
