datasetPath = fullfile('D:\CBIS_DDSM_PNG\masked_alexnet_272_272')


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

% Show labels
labelCount = countEachLabel(imds)

% Select Pretrained network for transfer learning
net = alexnet
% show layers
net.Layers
% AnalyzeNetwork(net)
inputSize = net.Layers(1).InputSize

%feature
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain);
augimdsTest = augmentedImageDatastore(inputSize(1:2),imdsTest);

layer = 'fc7';
featuresTrain = activations(net,augimdsTrain,layer,'OutputAs','rows');
featuresTest = activations(net,augimdsTest,layer,'OutputAs','rows');

% Extract class labels
YTrain = imdsTrain.Labels;
YTest = imdsTest.Labels;
% Fit image classifier
classifier = fitcecoc(featuresTrain,YTrain);
% Classify test image
YPred = predict(classifier,featuresTest);


%Show Accuracy 
accuracy = mean(YPred == YTest)

