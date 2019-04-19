% Guide: https://www.mathworks.com/help/deeplearning/ref/alexnet.html#bvn44n6
datasetPath = fullfile('F:\CBIS_DDSM_PNG\MASKED\Calc_Mask_v0_3_alexnet')
%datasetPath = fullfile('/Users/xfler/Documents/GitHub/Year4_FYP/Images/CBIS_DDSM_PNG/Calcification-Training/AlexNet_RGB/');
imds = imageDatastore(datasetPath,'IncludeSubfolders',true,'LabelSource','foldernames');
[imdsTrain,imdsTest] = splitEachLabel(imds,0.7,'randomized');
% Show Image
numTrainImages = numel(imdsTrain.Labels);
idx = randperm(numTrainImages,16);
figure
for i = 1:16
    subplot(4,4,i)
    I = readimage(imdsTrain,idx(i));
    imshow(I)
end

% Select alexnet
net = alexnet;
% Show inputsize
inputSize = net.Layers(1).InputSize
% Extract Image Features
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain);
augimdsTest = augmentedImageDatastore(inputSize(1:2),imdsTest);

layer = 'fc7';
featuresTrain = activations(net,augimdsTrain,layer,'OutputAs','rows');
featuresTest = activations(net,augimdsTest,layer,'OutputAs','rows');

% Extrat class labels
YTrain = imdsTrain.Labels;
YTest = imdsTest.Labels;
% Fit Image Classifier
classifier = fitcecoc(featuresTrain,YTrain);
% Classify Test Images
YPred = predict(classifier,featuresTest);
idx = [1 5 10 15];
figure
for i = 1:numel(idx)
    subplot(2,2,i)
    I = readimage(imdsTest,idx(i));
    label = YPred(idx(i));
    imshow(I)
    title(char(label))
end
accuracy = mean(YPred == YTest)
