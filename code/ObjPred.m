function YPred = ObjPred(x,YTrain_end,TNorm,mu,sig,Days,net)

%% Parameters
%% Taks Model
% Color noise parameters in social distanc/knowledge
Power_social = abs(x(1));
Power_know   = abs(x(2));

A_social = x(3);
B_social = x(4);

A_know = x(5);
B_know = x(6);

%% Facility Model

time_delay = abs(x(7)); % day

% Equalizer form
TL = x(8);
TI = x(9);
k_equalizer = x(10);

% Lag for Staff performance
TN_p = x(11);

% ======== ========  Staff model
% Staff frequency and damping ratio
zn_s = x(12);
wn_s = x(13);

% Human Senses
k_p_hs = x(14);
T_L_hs = x(15);
T_I_hs = x(16);

% Cognitive model
P_cog = x(17);
T_cog = abs(x(18));

% saturation
K = abs(x(19));

% ======== ===== Hospital model
% hospital performance
zn_h  = x(20);
wn_h  = x(21);
T_in  = abs(x(22));
T_emg = abs(x(23));
k_inp = x(24);

% Ambulance Entery
T_L_Amb = x(25);
T_I_Amb = x(26);

% Emergency Services
T_L_Emg = x(27);
T_I_Emg = x(28);

% Direct Medical Services
T_L_Med = x(29);
T_I_Med = x(30);

% Death TF
zn_de = x(31);
wn_de = x(32);

% Ambulance Power
Power_Amb = abs(x(33));

Power_Emg = abs(x(34));

Power_Med = abs(x(35));

Sequence_L = 1+round(9*abs(x(36)));
%% assign variable into workspace
% Taks Model
assignin('base','Power_social',Power_social);
assignin('base','Power_know',Power_know);

assignin('base','A_social',A_social);
assignin('base','B_social',B_social);

assignin('base','A_know',A_know);
assignin('base','B_know',B_know);

% Facility Model

assignin('base','time_delay',time_delay);

assignin('base','TL', TL);
assignin('base','TI', TI);
assignin('base','k_equalizer',k_equalizer);

assignin('base','TN_p', TN_p);

assignin('base','zn_s',zn_s);
assignin('base','wn_s',wn_s);

assignin('base','k_p_hs',k_p_hs);
assignin('base','T_L_hs',T_L_hs);
assignin('base','T_I_hs',T_I_hs);

assignin('base','P_cog',P_cog);
assignin('base','T_cog',T_cog);

assignin('base','K',K);

assignin('base','zn_h',zn_h);
assignin('base','wn_h',wn_h);
assignin('base','T_in',T_in);
assignin('base','T_emg',T_emg);
assignin('base','k_inp',k_inp);

assignin('base','T_L_Amb',T_L_Amb);
assignin('base','T_I_Amb',T_I_Amb);

assignin('base','T_L_Emg',T_L_Emg);
assignin('base','T_I_Emg',T_I_Emg);

assignin('base','T_L_Med',T_L_Med);
assignin('base','T_I_Med',T_I_Med);

assignin('base','zn_de',zn_de);
assignin('base','wn_de',wn_de);
%
assignin('base','Power_Amb',Power_Amb);
assignin('base','Power_Emg',Power_Emg);
assignin('base','Power_Med',Power_Med);

%
assignin('base','x',x);

assignin('base','net',net);

assignin('base','Days',Days);
assignin('base','YTrain_end',double(YTrain_end));
%
assignin('base','mu',mu);
assignin('base','sig',sig);
assignin('base','Sequence_L',Sequence_L)

%%x
warning off

Dyn_sim = sim('HybridSimulinkModel.slx','SimulationMode', 'accelerator');%,'TimeOut',Days);

Cases_sim  = Dyn_sim.Cases;
Deaths_sim =  Dyn_sim.Deaths;

data_sim = [Cases_sim Deaths_sim];

idx_sim = [];

parfor i = 1:Days
    idx_sim = [idx_sim find(Dyn_sim.tout == i)];
end

data_sim = data_sim(idx_sim,:);

[net,YPred] = predictAndUpdateState(net,YTrain_end');

for i = 2:Days
    
    [net,YPred(:,i)] = predictAndUpdateState(net,[YPred(:,i-1)+...
        data_sim(i,:)'],'ExecutionEnvironment','cpu','SequenceLength',Sequence_L);
    
    if i < size(TNorm,1)
        YPred(:,i) =  ((TNorm(i,:).*sig+mu)' + (YPred(:,i).*sig' + mu'))./2 ;
        YPred(:,i) = (YPred(:,i)- mu') ./ sig';
    else
        YPred(:,i) = YPred(:,i) + (mu'./sig').*rand(2,1);
    end
    %
    idx = (YPred(:,i).*sig' + mu')< 0;
    d = 0.01*rand(2,1);
    if sum(idx) > 0
        YPred(idx,i) = -mu(idx)'./sig(idx)' + d(idx);
    end
    
end


YPred = round(YPred.*sig' + mu');




end
