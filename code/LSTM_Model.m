clear
clc

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
C = [];
for ii=1:length(region)
    
    disp(region{ii})
    
    rmse = [];
    for j = 1:10
        
        Sequence_length = j
        
        %% import data
        
        file_name = 'dms.csv'; %
        opts = detectImportOptions(file_name,'NumHeaderLines',0);
        Data = readtable(file_name,opts);
        
        %% Extracting Countries
        id = strcmp(Data.countryterritoryCode,region{ii}); 
        
        Country = sortrows(Data(id,:),1);
        
        data_model = [Country.cases Country.deaths];        
        
        %%
        
        % figure
        %
        % subplot(2,1,1)
        % plot(data_model(:,1))
        % xlabel('Day')
        % ylabel('Cases')
        %
        % subplot(2,1,2)
        % plot(data_model(:,2))
        % xlabel('Day')
        % ylabel('Deaths')
        
        %% 90% train and 10% test data 
        numTimeStepsTrain = floor(0.9*size(data_model,1));
        
        dataTrain = data_model(1:numTimeStepsTrain+1,:);
        dataTest = data_model(numTimeStepsTrain+1:end,:);
        
        %%
        mu = mean(data_model);
        sig = std(data_model);
        
        dataTrainStandardized = (dataTrain - mu) ./ sig;
        
        %%
        XTrain = dataTrainStandardized(1:end-1,:);
        YTrain = dataTrainStandardized(2:end,:);
        
        %%
        
        numFeatures = 2;
        numResponses = 2;
        numHiddenUnits = 400;
        
        layers = [
            sequenceInputLayer(numFeatures)
            lstmLayer(numHiddenUnits)
            fullyConnectedLayer(numResponses)
            regressionLayer
            ];
        %%
        options = trainingOptions('sgdm', ...
            'Momentum',0.95, ...
            'MaxEpochs',500, ...
            'GradientThreshold',1, ...
            'InitialLearnRate',0.01, ...
            'LearnRateSchedule','piecewise', ...
            'LearnRateDropPeriod',500, ...
            'LearnRateDropFactor',0.2, ...
            'SequenceLength',Sequence_length,...
            'Verbose',1, ...
            'ExecutionEnvironment','cpu',... % parallel cpu
            'Plots','none'); %'training-progress'
        %%
        net = trainNetwork(XTrain',YTrain',layers,options);
        % load('net.mat');
        save(['net_' region{ii} '_' num2str(Sequence_length)],'net');
        
        %%
        dataTestStandardized = (dataTest - mu) ./ sig;
        XTest = dataTestStandardized(1:end-1,:);
        
        %%
        net = predictAndUpdateState(net,XTrain');
        [net,YPred] = predictAndUpdateState(net,YTrain(end,:)');
        
        numTimeStepsTest = size(XTest,1);
        for i = 2:numTimeStepsTest
            [net,YPred(:,i)] = predictAndUpdateState(net,YPred(:,i-1),...
                'SequenceLength',Sequence_length,'ExecutionEnvironment','cpu'); % 'SequenceLength',10
        end
        
        %%
        YPred = YPred.*sig' + mu';
%         YPred = round(YPred);
        
        %%
        YTest = dataTest(2:end,:);
        
        rmse = [rmse;sqrt(mean((YPred'-YTest).^2))];
        rmse(end,:)
        C = [C;{region Sequence_length rmse(end,1) rmse(end,2)}];
    end
    RMSE{ii} = rmse;
    
end
T = cell2table(C,...
    'VariableNames',{'Region' 'Sequence_length' 'rmse_Cases' 'rmse_Deaths'});

save('RMSE_LSTM_LSequence','RMSE');
save('T_LSTM_LSequence','T');

