clc
clear

%%
% add Net and Optimized Parameters as directory
path_1 = '...';
path_2 = '...';

addpath(path_1);
addpath(path_2);
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


area = {'USA'};


%%
for jj = 1:length(area)
    
    region = area{jj};
    
    Sequence_length = 10;
    
    net_name = ['net_' region '_' num2str(Sequence_length) '.mat'];
    
    Net = load(net_name);
    net = Net.net;
    load(['Xopt_' region '_'  num2str(Sequence_length) '.mat'])
    
    %% import data
    
    file_name = 'dms.csv'; %
    opts = detectImportOptions(file_name,'NumHeaderLines',0);
    Data = readtable(file_name,opts);
    
    %% Extracting Countries
    id = strcmp(Data.countryterritoryCode,region); % United_States_of_America
    
    Country = sortrows(Data(id,:),1);
    
    data_model = [Country.cases Country.deaths];
    
    date_rep = Country.dateRep;
    
    %%
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
    dataTestStandardized = (dataTest - mu) ./ sig;
    XTest = dataTestStandardized(1:end-1,:);
    YTest = dataTestStandardized(2:end,:).*sig + mu;
    
    dataModelStandard = (data_model - mu) ./ sig;
    
    %% Hybrid Model
    numTimeStepsTest = size(YTest,1);
    Days = numTimeStepsTest;
    
    [net_h,YPred] = predictAndUpdateState(net,YTrain(end,:)');
    
    YPred_hybrid = ObjPred(x,YTrain(end,:),dataTestStandardized(2:end,:),mu,sig,Days,net);
    
    rmse_hybrid = sqrt(mean((YPred_hybrid'-YTest).^2));
    
    %% LSTM
    YPred_lstm = YPred;
    for i = 2:numTimeStepsTest
        
        [net_h,YPred_lstm(:,i)] = predictAndUpdateState(net_h,YPred_lstm(:,i-1)...
            ,'ExecutionEnvironment','cpu','SequenceLength',Sequence_length);
    end
    
    YPred_lstm = round(YPred_lstm.*sig' + mu');
    rmse_lstm = sqrt(mean((YPred_lstm'-YTest).^2));
    
    %% Future Prediction
    
    future_day = 30;
    
    
    [net_f,YPred] = predictAndUpdateState(net,YTrain(end,:)');
    % net_f = net;
    for i = 1:numTimeStepsTest
        [net_f,YPred_f(:,i)] = predictAndUpdateState(net_f,XTest(i,:)'...
            ,'ExecutionEnvironment','cpu');
    end
    
    % lstm
    YFuture_lstm = YPred_f(:,end);
    net_lstm = net_f;
    for i = 2:future_day
        [net_lstm,YFuture_lstm(:,i)] = predictAndUpdateState(net_lstm,YFuture_lstm(:,i-1)...
            ,'ExecutionEnvironment','cpu');
    end
    YFuture_lstm = round(YFuture_lstm.*sig' + mu');
    
    YFuture_hybrid = ObjPred(x,YTest(end,:),dataTestStandardized(2:end,:),mu,sig,future_day,net);
    
    %% Cases
    figure;
    subplot(2,1,1)
    box on
    
    s_train = size(dataTrain,1);
    s_test = size(dataTest,1);
    
    idx1 = date_rep(1:s_train-1);
    idx2 = date_rep(s_train:s_train+s_test-1);
    idx3 = date_rep(end):date_rep(end)+days(future_day);
    
    b = [dataTrain;dataTest;YPred_lstm';YPred_hybrid';YFuture_lstm';YFuture_hybrid'];
    
    
    hold on
    
    plot([idx1;idx2(2)],dataTrain(1:end,1),'k','LineWidth',1.2)
    
    % Test Figure
    plot(idx2,[data_model(s_train,1);YTest(:,1)],'g','LineWidth',1.2)
    plot(idx2,[data_model(s_train,1);YPred_lstm(1,:)'],'.-b','LineWidth',1.2)
    plot(idx2,[data_model(s_train,1);YPred_hybrid(1,:)'],'o-r','LineWidth',1.2)
    
    % future plots
    plot(idx3,[YPred_lstm(1,end)';YFuture_lstm(1,:)'],'.-b','LineWidth',1.2)
    plot(idx3,[YPred_hybrid(1,end)';YFuture_hybrid(1,:)'],'o-r','LineWidth',1.2,'MarkerSize',5)
    
    % Seperation Line
    plot([idx2(1) idx2(1)],[0 max(b(:,1))+10],'--k','LineWidth',1)
    plot([idx2(end) idx2(end)],[0 max(b(:,1))+10],'--k','LineWidth',1)
    
    
    
    hold off
    xlabel('Day')
    ylabel([region ' (Cases)'])
    title(['RMSE: LSTM = ' num2str(rmse_lstm(1)) ', Hybrid Model = ' num2str(rmse_hybrid(1))]);
    legend(["Observed" "Test" "LSTM" "Hybrid Model"],'Location','northwest','NumColumns',1)
    
    %% Change figures font
    fh = findall(0,'Type','Figure');
    txt_obj = findall(fh,'Type','text');
    
    set(txt_obj,'FontName','Times New Roman','FontSize',10)
    
    %% Deaths
    subplot(2,1,2);
    box on
    hold on
    
    plot([idx1;idx2(1)],dataTrain(1:end,2),'k','LineWidth',1.2)
    
    % Test Figure
    plot(idx2,[data_model(s_train,2);YTest(:,2)],'g','LineWidth',1.2)
    plot(idx2,[data_model(s_train,2);YPred_lstm(2,:)'],'.-b','LineWidth',1.2)
    plot(idx2,[data_model(s_train,2);YPred_hybrid(2,:)'],'o-r','LineWidth',1.2)
    
    % future plots
    plot(idx3,[YPred_lstm(2,end)';YFuture_lstm(2,:)'],'.-b','LineWidth',1.2)
    plot(idx3,[YPred_hybrid(2,end)';YFuture_hybrid(2,:)'],'o-r','LineWidth',1.2,'MarkerSize',5)
    
    % Seperation Line
    plot([idx2(1) idx2(1)],[0 max(b(:,2))+10],'--k','LineWidth',1)
    plot([idx2(end) idx2(end)],[0 max(b(:,2))+10],'--k','LineWidth',1)
    
    hold off
    xlabel('Day')
    ylabel([region ' (Deaths)'])
    title(['RMSE: LSTM = ' num2str(rmse_lstm(2)) ', Hybrid Model = ' num2str(rmse_hybrid(2))])
    legend(["Observed" "Test" "LSTM" "Hybrid Model"],'Location','northwest','NumColumns',1)
    
    %% Change figures font
    fh = findall(0,'Type','Figure');
    txt_obj = findall(fh,'Type','text');
    
    set(txt_obj,'FontName','Times New Roman','FontSize',10)
end