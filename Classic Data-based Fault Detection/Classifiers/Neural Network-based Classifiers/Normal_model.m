%% NARX(MLP)
clc;close all;clear;
%%
load data_test
load data_train
%% param NN
Nern_Num_Lim=[3 15];
Nern_Step=3;
Epochs =50;

%% dadehaye voroodi
% normal data
X_nrml=[u;[0,u(1,1:999)];[0,0,0,u(1,1:997)];[0,y_n(1,1:999)];[0,0,y_n(1,1:998)];[0,0,0,y_n(1,1:997)]];
Y_nrml=y_n;

a1=randperm(1000);
X_train_nrml=X_nrml(:,a1(1:700));
Y_train_nrml=Y_nrml(:,a1(1:700));

X_test_nrml=X_nrml(:,a1(701:end));
Y_test_nrml=Y_nrml(:,a1(701:end));

% normalize
Min=min(X_train_nrml');
h=max(X_train_nrml')-min(X_train_nrml');

Min1=min(Y_train_nrml');
h1=max(Y_train_nrml')-min(Y_train_nrml');

X_train_n_Normal=(((X_train_nrml'-ones(700,1)*Min)./(ones(700,1)*h))*2-1)';
X_test_n_Normal=(((X_test_nrml'-ones(300,1)*Min)./(ones(300,1)*h))*2-1)';

Y_train_n_Normal=(((Y_train_nrml'-ones(700,1)*Min1)./(ones(700,1)*h1))*2-1)';
Y_test_n_Normal=(((Y_test_nrml'-ones(300,1)*Min1)./(ones(300,1)*h1))*2-1)';

%% NN for normal data
count=0;
for KN=Nern_Num_Lim(1):Nern_Step:Nern_Num_Lim(2)  
    count=count+1;  
    net=newff(X_train_n_Normal,Y_train_n_Normal,KN,{'tansig' 'purelin'});   
    net=init(net);
    net.trainFcn = 'trainlm';
    net.trainParam.epochs = 1;    
    SSE_TR=[];
    SSE_TE=[];
    for i=1:Epochs
        [net,tr]=train(net,X_train_n_Normal,Y_train_n_Normal);
        SSE_TR_a = sum( (Y_train_n_Normal-sim(net,X_train_n_Normal)).^2 );
        SSE_TE_a = sum( (Y_test_n_Normal-sim(net,X_test_n_Normal)).^2 );
        SSE_TR = [SSE_TR SSE_TR_a];
        SSE_TE  =[SSE_TE SSE_TE_a];
        
        R=randperm(700);
        X_train_n_Normal=X_train_n_Normal(:,R);
        Y_train_n_Normal=Y_train_n_Normal(:,R);   
    end
    Out_TR = sim(net,X_train_n_Normal);
    E_TR=Out_TR-Y_train_n_Normal;
    MSE_TR(count)=mse(E_TR); 
    Out_TE = sim(net,X_test_n_Normal);
    E_TE=Out_TE-Y_test_n_Normal;
    MSE_TE(count)=mse(E_TE);
end
%%
figure
plot((Nern_Num_Lim(1):Nern_Step:Nern_Num_Lim(2)),MSE_TE(1,:,end),'r')
hold on
plot((Nern_Num_Lim(1):Nern_Step:Nern_Num_Lim(2)),MSE_TR(1,:,end))
grid on
xlabel('Neuron')
ylabel('MSE')
title('MSE Per Neuron')
legend ('Test','Train')


figure
plot(1:Epochs,SSE_TE/300,'r')
hold on
plot(1:Epochs,SSE_TR/700)
grid on
xlabel('Epochs')
ylabel('MSE')
title('MSE Per Epochs For optimum neuron')
legend ('Test','Train')


Out_TE = sim(net,X_test_n_Normal);
figure
plot(Y_test_n_Normal)
hold on
plot(Out_TE,'r--')
grid on
title('y & y_h_a_t For Test')
legend ('y','y_h_a_t')
axis([100 200 -1 1])

fitting_percent=(1-(norm(Y_test_n_Normal-(Out_TE))/norm(Y_test_n_Normal-mean(Y_test_n_Normal))))*100