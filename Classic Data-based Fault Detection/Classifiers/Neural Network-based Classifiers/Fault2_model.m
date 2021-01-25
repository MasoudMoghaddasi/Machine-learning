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
% fault2 data
X_f2=[[0,u(1,1:999)];[0,0,y_f2(1,1:998)]];
Y_f2=y_f2;

a1=randperm(1000);
X_train_f2=X_f2(:,a1(1:700));
Y_train_f2=Y_f2(:,a1(1:700));

X_test_f2=X_f2(:,a1(701:end));
Y_test_f2=Y_f2(:,a1(701:end));

% normalize
Min=min(X_train_f2');
h=max(X_train_f2')-min(X_train_f2');

Min1=min(Y_train_f2');
h1=max(Y_train_f2')-min(Y_train_f2');

X_train_f2_Normal=(((X_train_f2'-ones(700,1)*Min)./(ones(700,1)*h))*2-1)';
X_test_f2_Normal=(((X_test_f2'-ones(300,1)*Min)./(ones(300,1)*h))*2-1)';

Y_train_f2_Normal=(((Y_train_f2'-ones(700,1)*Min1)./(ones(700,1)*h1))*2-1)';
Y_test_f2_Normal=(((Y_test_f2'-ones(300,1)*Min1)./(ones(300,1)*h1))*2-1)';

%% NN for fault2 data
count=0;
for KN=Nern_Num_Lim(1):Nern_Step:Nern_Num_Lim(2)  
    count=count+1;  
    net=newff(X_train_f2_Normal,Y_train_f2_Normal,KN,{'tansig' 'purelin'});   
    net=init(net);
    net.trainFcn = 'trainlm';
    net.trainParam.epochs = 1;    
    SSE_TR=[];
    SSE_TE=[];
    for i=1:Epochs
        [net,tr]=train(net,X_train_f2_Normal,Y_train_f2_Normal);
        SSE_TR_a = sum( (Y_train_f2_Normal-sim(net,X_train_f2_Normal)).^2 );
        SSE_TE_a = sum( (Y_test_f2_Normal-sim(net,X_test_f2_Normal)).^2 );
        SSE_TR = [SSE_TR SSE_TR_a];
        SSE_TE  =[SSE_TE SSE_TE_a];
        
        R=randperm(700);
        X_train_f2_Normal=X_train_f2_Normal(:,R);
        Y_train_f2_Normal=Y_train_f2_Normal(:,R);   
    end
    Out_TR = sim(net,X_train_f2_Normal);
    E_TR=Out_TR-Y_train_f2_Normal;
    MSE_TR(count)=mse(E_TR); 
    Out_TE = sim(net,X_test_f2_Normal);
    E_TE=Out_TE-Y_test_f2_Normal;
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


Out_TE = sim(net,X_test_f2_Normal);
figure
plot(Y_test_f2_Normal)
hold on
plot(Out_TE,'r--')
grid on
title('y & y_h_a_t For Test')
legend ('y','y_h_a_t')
axis([100 200 -1 1])

fitting_percent=(1-(norm(Y_test_f2_Normal-(Out_TE))/norm(Y_test_f2_Normal-mean(Y_test_f2_Normal))))*100