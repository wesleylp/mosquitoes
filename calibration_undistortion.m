%Coppe/UFRJ
%Wesley Lobato Passos
%calibração de Zhang e distorção radial

clear; close all; clc;
addpath('utils');
fprintf('\n Coppe/UFRJ');
fprintf('\n Wesley Lobato Passos');
fprintf('\n Zhang calibration and Radial undistortion');
%warning('OFF');
mkdir('figures');
pause;

%% 1 Reading the files and showing the images
fprintf('\n 1 Reading and displaying... \n');

for k = 1:9
    imgDistName{k} = sprintf('../../Public/cam_calibration/image_dist_%d.jpg',k);
    imgDist{k} = im2double(imread(imgDistName{k}));
    figure('Name',imgDistName{k},'NumberTitle','off')
    imshow(imgDist{k});
                
    %figure settings
    position = [50 100 50+size(imgDist{k},2) 100+size(imgDist{k},1)];
    figProp = struct('size',26,'font','Times','lineWidth',1.6,'figDim',position);
    %saving figure...    
    figName=sprintf('image_dist%d',k');
    figFileName = sprintf('figures/%s',figName);
    %formatFig(gcf,figFileName,'en',figProp);
    
    pause;
    
    
end
close all;
    
%% 2 Detecting keypoints
fprintf('\n 2 Detecting keypoints... \n');
tic;
[keyDistPts,boardDistSize] = detectCheckerboardPoints(imgDistName);    
fprintf('time: %4.4f',toc);
pause;

%% 3 Camera calibration
fprintf('\n 3 Camera calibration... \n');

for k=1:9
    figure('Name',imgDistName{k},'NumberTitle','off')
    imshow(imgDist{k})
    hold on
    plot(keyDistPts(:,1,k),keyDistPts(:,2,k),'.r','MarkerSize',16)    
    
    %figure settings
    position = [50 100 50+size(imgDist{k},2) 100+size(imgDist{k},1)];
    figProp = struct('size',26,'font','Times','lineWidth',1.6,'figDim',position);
    %saving figure...
    figName=sprintf('image_dist_pts%d',k');
    figFileName = sprintf('figures/%s',figName);
    %formatFig(gcf,figFileName,'en',figProp);
    
    pause;        
end
close all;

% 4 Real World points
fprintf('\n Real world points... \n');
squareSize = 29; %in mm
worldDistPts = generateCheckerboardPoints(boardDistSize, squareSize);
pause;

% 5 Camera Params
fprintf('\n Obtaining Camera Parameters... \n');
[cameraDistParams,~, estimationDistErrors] = estimateCameraParameters(keyDistPts,worldDistPts);
pause;

% 6 Showing extrinsics cam params
fprintf('\n Vizualizing extrinsic parameters... \n');

%Visualize pattern locations
figure('Name','Extrisic Params: pattern locations' ,'NumberTitle','off')
showExtrinsics(cameraDistParams);

title('');
xlabel('$x (mm)$','interpreter','latex')
ylabel('$z (mm)$','interpreter','latex')
zlabel('$y (mm)$','interpreter','latex')

%figure settings
position = [1 1 920 720];
figProp = struct('size',26,'font','Times','lineWidth',1.6,'figDim',position);
%saving figure...
figName=sprintf('Extrisics_part3_6');
figFileName = sprintf('figures/%s',figName);
%formatFig(gcf,figFileName,'en',figProp);

pause;


%Visualize camera locations
figure('Name','Extrisic Params: cam locations' ,'NumberTitle','off');
showExtrinsics(cameraDistParams,'patternCentric');

title('');
xlabel('$x (mm)$','interpreter','latex')
ylabel('$y (mm)$','interpreter','latex')
zlabel('$z (mm)$','interpreter','latex')

%figure settings
position = [1 1 920 720];
figProp = struct('size',26,'font','Times','lineWidth',1.6,'figDim',position);
%saving figure...
figName=sprintf('Extrisics_part3_6PatternCentric');
figFileName = sprintf('figures/%s',figName);
%formatFig(gcf,figFileName,'en',figProp);

pause;

%7 Camera intrisics Params
fprintf('\n Camera intrisics Params... \n');
KDist = cameraDistParams.IntrinsicMatrix;
fprintf('\n Intrinsic matrix... \n');
disp(KDist.');

fprintf('\n Distância focal (pixels): [%4.6f %4.6f]\n',...
    cameraDistParams.FocalLength(1),cameraDistParams.FocalLength(2));
fprintf('\n Ponto Principal (pixels): [%4.6f %4.6f]\n',...
    cameraDistParams.PrincipalPoint(1),cameraDistParams.PrincipalPoint(2));
fprintf('\n Obliquidade (skew): [%4.6f]\n',...
    cameraDistParams.Skew);
pause;


% 8 Evaluating the accuracy camera calibration
fprintf('\n Evaluating the accuracy camera calibration... \n');

figure('Name','Reprojection erros' ,'NumberTitle','off')
showReprojectionErrors(cameraDistParams);

title('');
xlabel('Imagens','interpreter','latex')
ylabel('Erro m\''{e}dio (em pixels)','interpreter','latex')
lg1 = legend('hide');

%figure settings
position = [1 1 920 720];
figProp = struct('size',26,'font','Times','lineWidth',1.6,'figDim',position);
%saving figure...
figName=sprintf('reprojectionError_part3_6');
figFileName = sprintf('figures/%s',figName);
%formatFig(gcf,figFileName,'en',figProp);

displayErrors(estimationDistErrors,cameraDistParams)
pause;

%% 4 Correct Lens Distortion
fprintf('\n 4 Correct Lens Distortion... \n');

for k=1:10
    imgUndist{k} = undistortImage(imgDist{k},cameraDistParams);   
    
    figure('Name',imgDistName{k},'NumberTitle','off')
    imshowpair(imgDist{k}, imgUndist{k}, 'montage');
    title('Imagem Distorcida (equerda) e Imagem Corrigida (direita)');  
    
    %figure settings
    position = [50 100 50+size(imgDist{k},2)+size(imgUndist{k},2) 100+size(imgDist{k},1)];
    figProp = struct('size',26,'font','Times','lineWidth',1.6,'figDim',position);
    %saving figure...
    figName=sprintf('image_dist_comp%d',k');
    figFileName = sprintf('figures/%s',figName);
    %formatFig(gcf,figFileName,'en',figProp);
    
    pause;
    
    
    imgUndistFull{k} = undistortImage(imgDist{k}, cameraDistParams, 'OutputView', 'full');    
    figure('Name',imgDistName{k},'NumberTitle','off')
    imshow(imgUndistFull{k})
    title('Imagem Corrigida completa');
    
    
    %figure settings
    position = [50 100 50+size(imgUndistFull{k},2) 100+size(imgUndistFull{k},1)];
    figProp = struct('size',26,'font','Times','lineWidth',1.6,'figDim',position);
    %saving figure...
    figName=sprintf('image_dist_correct%d',k');
    figFileName = sprintf('figures/%s',figName);
    %formatFig(gcf,figFileName,'en',figProp);
        
    pause;    
end
close all;

%%
fprintf('\n\n =============FIM!============= \n');
pause;
close all; clear; clc;
