%%
clc
clear all
close all
% Weight_Matrix for image recovery
% Author: Tianrui Zhao King's College London
% 07/08/2019
DMD = 64; % change the input pattern size
Input = hadamard(DMD^2);  % [img_size,img_size,Number = img_size^2x2]
Input1 = single([Input > 0]);
Input2 = ones(DMD^2,DMD^2) - Input1;
Input = single([Input1,Input2]); 

%% Import Output speckles
tic
imgpath = 'Speckle'; % load data path
M = DMD^2*2;% The number of measurements
clear D;
for i = 1:M
hadamard_A_path = sprintf('%s\\Image0_%d.tif',imgpath,i);
A = imread(hadamard_A_path);
D(:,:,i) = A;
end

B = double(D(:));
[a,b] = size(D(:,:,1));
Output = single(reshape(B,a*b,M));

sprintf('Data-load finished')
toc

R = double(D(:,:,1));
Outputdiff = 2*Output - R(:);
Measure = M;
Inputhada = 2*Input - ones(DMD*DMD,Measure);

sprintf('Starting ITM')

tic
W = (Outputdiff*Inputhada')./Measure;
WN = pinv(W);
toc

sprintf('Starting image recovery')
%%
Q = size(D,3);
Image_allg = zeros(DMD*DMD,Q);
tic
clear CO;
for t = 1:Q
    dirinfo = sprintf('%s\\Image1_%d.tif',imgpath,t);% speckle path
    Test1 = imread(dirinfo);
    Signal = double(Test1);
    tic
    Image = WN*Signal(:);
    Image = reshape(Image,DMD,DMD);
    toc
    imshow(Image,[])
    label_path = sprintf('%s\\Data_%d.png',imgpath,t);% save path
    imwrite(Image,label_path)
end
toc