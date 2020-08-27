clc
clear all
%% Initialization
% United_states_of_America: USA
% Brazil: BRA
% India: IND
% Russia: RUS
% South Africa: ZAF
% Mexico: MEX
% Peru: PER
% Colombia: COL
% Chile: CHL
% Iran:  IRN
% Australia: AUS

region = {'USA','BRA','IND','RUS','ZAF','MEX','PER','COL','CHL','IRN','AUS'};
RMSE_RNN = []; 

for i = 1:length(region)
    %% import data
    
    file_name = 'dms.csv'; %
    opts = detectImportOptions(file_name,'NumHeaderLines',0);
    Data = readtable(file_name,opts);
    
    %% Extracting Countries
    id = strcmp(Data.countryterritoryCode,region{i}); % United_States_of_America
    
    Country = sortrows(Data(id,:),1);
    
    data_model = [Country.cases Country.deaths];
    
    
    %% 90% Train , 10% Test
    numTimeStepsTrain = floor(0.9*size(data_model,1));
    
    dataTrain = data_model(1:numTimeStepsTrain+1,:);
    dataTest = data_model(numTimeStepsTrain+1:end,:);
    
    %%
    mu = mean(data_model);
    sig = std(data_model);
    
    dataTrainStandardized = (dataTrain - mu) ./ sig;
    
    %% Train
    XTrain = dataTrainStandardized(1:end-1,:);
    YTrain = dataTrainStandardized(2:end,:);
    
    %% Test
    dataTestStandardized = (dataTest - mu) ./ sig;
    XTest = dataTestStandardized(1:end-1,:);
    YTest = dataTestStandardized(2:end,:).*sig + mu;
    
    %% Cases Training
    net = layrecnet(7,20);
    net.trainParam.epochs = 500;
    
    [Xs,Xi,Ai,Ts] = preparets(net,num2cell(XTrain(:,1)'),num2cell(YTrain(:,1)'));
    net = train(net,Xs,Ts,Xi,Ai,'useParallel','yes');
   
    
    Y = net(num2cell(XTest(:,1)'),Xi,Ai);
    YPred = cell2mat(Y);
    YPred = YPred.*sig(1)+mu(1);
    
    rmse_cases = sqrt(mean((YPred'-YTest(:,1)).^2));
    
    %% Deaths Training
    
    net = layrecnet(7,20);
    net.trainParam.epochs = 500;
    
    [Xs,Xi,Ai,Ts] = preparets(net,num2cell(XTrain(:,2)'),num2cell(YTrain(:,2)'));
    net = train(net,Xs,Ts,Xi,Ai,'useParallel','yes');
    
    %     view(net)
    
    Y = net(num2cell(XTest(:,2)'),Xi,Ai);
    YPred = cell2mat(Y);
    YPred = YPred.*sig(2)+mu(2);
    
    rmse_deaths = sqrt(mean((YPred'-YTest(:,1)).^2));
    
    RMSE_RNN = [RMSE_RNN;rmse_cases rmse_deaths]
    
    
end

save('RMSE_RNN','RMSE_RNN');
