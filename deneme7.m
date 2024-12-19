% Eğitim ve test veri setlerini yükle
trainData = imageDatastore('train', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
testData = imageDatastore('test', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% Görüntüleri  boyutlandır
inputSize = [150 150];
trainData.ReadFcn = @(filename) preprocessImage(filename, inputSize);
testData.ReadFcn = @(filename) preprocessImage(filename, inputSize);

function img = preprocessImage(filename, inputSize)
    img = imread(filename);
    if size(img, 3) == 3 % RGB kontrolü
        img = rgb2gray(img);
    end
    img = imresize(img, inputSize); % Boyutlandırma
    img = im2double(img); % Normalize etme
end

% SMOTE ile sınıf dengesini artırma (NORMAL sınıfı için)
labels = trainData.Labels;
normalIdx = labels == 'NORMAL';
pneumoniaIdx = labels == 'PNEUMONIA';

% SMOTE uygulaması
if sum(normalIdx) < sum(pneumoniaIdx)
    numToAdd = sum(pneumoniaIdx) - sum(normalIdx); % Eksik olan NORMAL sınıflar
    normalFiles = trainData.Files(normalIdx);
    syntheticFiles = datasample(normalFiles, numToAdd, 'Replace', true); % Oversampling
    for i = 1:numToAdd
        [~, fileName, ext] = fileparts(syntheticFiles{i});
        newFileName = fullfile('train', 'NORMAL', [fileName '_synthetic' num2str(i) ext]);
        copyfile(syntheticFiles{i}, newFileName);
    end
    % Veri setini yeniden yükle
    trainData = imageDatastore('train', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
    trainData.ReadFcn = @(filename) preprocessImage(filename, inputSize);
end

% Data augmentation için imageDataAugmenter
augmenter = imageDataAugmenter( ...
    'RandRotation', [-10, 10], ...
    'RandXTranslation', [-5, 5], ...
    'RandYTranslation', [-5, 5], ...
    'RandXScale', [0.9, 1.1], ...
    'RandYScale', [0.9, 1.1]);

augmentedTrainData = augmentedImageDatastore(inputSize, trainData, 'DataAugmentation', augmenter);

% Eğitim ve doğrulama veri setlerini böl
[trainData, valData] = splitEachLabel(trainData, 0.8, 'randomize');

% CNN modeli
layers = [
    imageInputLayer([150 150 1], 'Name', 'input')

    convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv1')
    batchNormalizationLayer('Name', 'batchnorm1')
    reluLayer('Name', 'relu1')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool1')

    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv2')
    batchNormalizationLayer('Name', 'batchnorm2')
    reluLayer('Name', 'relu2')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool2')

    convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'conv3')
    batchNormalizationLayer('Name', 'batchnorm3')
    reluLayer('Name', 'relu3')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool3')

    fullyConnectedLayer(256, 'Name', 'fc1')
    reluLayer('Name', 'relu4')
    dropoutLayer(0.5, 'Name', 'dropout1')

    fullyConnectedLayer(2, 'Name', 'fc2')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')
];

% Eğitim seçenekleri (erken durma eklenmiş)
options = trainingOptions('adam', ...
    'MiniBatchSize', 32, ...
    'MaxEpochs', 15, ...
    'InitialLearnRate', 1e-4, ...
    'ValidationData', valData, ...
    'ValidationFrequency', 10, ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'OutputFcn', @(info) stopIfAccuracyNotImproving(info, 5)); % Erken durma

% Modeli eğit
trainedNet = trainNetwork(trainData, layers, options);

% Modeli kaydetme
modelFileName = 'pneumonia_detection_model_D7.mat'; % Kaydedilecek dosya adı
save(modelFileName, 'trainedNet');
disp(['Model başarıyla kaydedildi: ', modelFileName]);

% Test veri setinde doğruluk hesapla
predictedLabels = classify(trainedNet, testData);
actualLabels = testData.Labels;

accuracy = mean(predictedLabels == actualLabels);
disp("Test doğruluğu: " + accuracy);

% F1 Skor ve MSE hesaplama
actualBinary = double(actualLabels == 'PNEUMONIA'); 
predictedBinary = double(predictedLabels == 'PNEUMONIA');

mse = mean((actualBinary - predictedBinary).^2);
disp("MSE: " + mse);

tp = sum((predictedBinary == 1) & (actualBinary == 1));
fp = sum((predictedBinary == 1) & (actualBinary == 0));
fn = sum((predictedBinary == 0) & (actualBinary == 1));

precision = tp / (tp + fp);
recall = tp / (tp + fn);
f1Score = 2 * (precision * recall) / (precision + recall);
disp("F1 Skor: " + f1Score);

% Karışıklık Matrisi
figure;
confusionchart(actualLabels, predictedLabels, ...
    'Title', 'Confusion Matrix', ...
    'RowSummary', 'row-normalized', ...
    'ColumnSummary', 'column-normalized');

% ROC Eğrisi
[~, scores] = max(predict(trainedNet, testData), [], 2);
[X, Y, T, AUC] = perfcurve(actualLabels, scores, 'PNEUMONIA');
figure;
plot(X, Y);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title(sprintf('ROC Curve (AUC = %.2f)', AUC));

% Yeni bir görüntüde pnömoni testi yapma
[file, path] = uigetfile({'*.png;*.jpg;*.jpeg', 'Görüntü Dosyaları (*.png, *.jpg, *.jpeg)'; '*.*', 'Tüm Dosyalar (*.*)'}, 'Pnömoni Test Görüntüsü Seçin');
if isequal(file, 0)
    disp('Görüntü seçimi iptal edildi.');
else
    % Seçilen görüntünün tam yolunu oluştur
    testImagePath = fullfile(path, file);

    % Görüntüyü ön işleme (preprocessing) ile hazırla
    testImage = preprocessImage(testImagePath, inputSize);

    % Görüntüyü modele sınıflandırma için gönder
    testImageLabel = classify(trainedNet, testImage);

    % Sonucu görüntüle
    disp(['Seçilen görüntü pnömoni teşhisi için sınıflandırıldı: ', char(testImageLabel)]);
    
    % Görüntüyü göster ve tahmin edilen etiketi yaz
    figure;
    imshow(testImage, []);
    title(['Tahmin: ', char(testImageLabel)]);
end

% Erken durma fonksiyonu
function stop = stopIfAccuracyNotImproving(info, N)
    stop = false;
    persistent bestValAccuracy
    persistent valLag

    if info.State == "start"
        bestValAccuracy = 0;
        valLag = 0;
    elseif info.State == "iteration"
        if ~isempty(info.ValidationAccuracy)
            if info.ValidationAccuracy > bestValAccuracy
                bestValAccuracy = info.ValidationAccuracy;
                valLag = 0;
            else
                valLag = valLag + 1;
            end
        end
        if valLag >= N
            stop = true;
        end
    end
end
