load cbis_ddsm_alexnet
% Guide: https://www.mathworks.com/help/deeplearning/ref/alexnet.html#bvn44n6
datasetPath = fullfile('F:\CBIS_DDSM_PNG\MASKED\Calc_Mask_v0_3_alexnet');
%datasetPath = fullfile('/Users/xfler/Documents/GitHub/Year4_FYP/Images/CBIS_DDSM_PNG/Calcification-Training/AlexNet_RGB/');
imds = imageDatastore(datasetPath,'IncludeSubfolders',true,'LabelSource','foldernames');
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7,'randomized');

% Show Image
numTrainImages = numel(imdsTrain.Labels);
idx = randperm(numTrainImages,16);
figure
for i = 1:16
    subplot(4,4,i)
    I = readimage(imdsTrain,idx(i));
    imshow(I)
end

% Show labels
labelCount = countEachLabel(imds);

% Select Pretrained network for transfer learning
% Others: https://www.mathworks.com/help/deeplearning/ug/pretrained-convolutional-neural-networks.html
net = cbis_ddsm_alexnet;

[YPred,scores] = classify(net,imdsValidation);
idx = randperm(numel(imdsValidation.Files));
figure
for i = 1:25 
    subplot(5,5,i)
    I = readimage(imdsValidation,idx(i));
    imshow(I)
    label = YPred(idx(i));
    title(string(label));
end


T = table(imdsValidation.files,YPred);

YValidation = imdsValidation.Labels;
accuracy = mean(YPred == YValidation);

filename = 'patientdata.xlsx';
writetable(T,filename,'Sheet',1)
