clc;close all;clear;
%%
load data_test
load data_train
%% NN's parameters
Nn=9;
Epochs_n=8;

Nf1=12;
Epochs_f1=10;

Nf2=10;
Epochs_f2=13;

%% Normal
% normal data
X_nrml=[u;[0,u(1,1:999)];[0,0,0,u(1,1:997)];[0,y_n(1,1:999)];[0,0,y_n(1,1:998)];[0,0,0,y_n(1,1:997)]];
Y_nrml=y_n;
% normalize
MinxN=min(X_nrml');
hxN=max(X_nrml')-min(X_nrml');
MinyN=min(Y_nrml');
hyN=max(Y_nrml')-min(Y_nrml');
X_train_n_Normal=(((X_nrml'-ones(1000,1)*MinxN)./(ones(1000,1)*hxN))*2-1)';
Y_train_n_Normal=(((Y_nrml'-ones(1000,1)*MinyN)./(ones(1000,1)*hyN))*2-1)';
% NN for normal data  
netN=newff(X_train_n_Normal,Y_train_n_Normal,Nn,{'tansig' 'purelin'});   
netN=init(netN);
netN.trainFcn = 'trainlm';
netN.trainParam.epochs = 1;    
for i=1:Epochs_n
    [netN,trN]=train(netN,X_train_n_Normal,Y_train_n_Normal);
    R=randperm(1000);
    X_train_n_Normal=X_train_n_Normal(:,R);
    Y_train_n_Normal=Y_train_n_Normal(:,R);   
end

%% fault1
% fault1 data
X_f1=[u;[0,u(1,1:999)];[0,0,y_f1(1,1:998)]];
Y_f1=y_f1;
% normalize
Minxf1=min(X_f1');
hxf1=max(X_f1')-min(X_f1');
Minyf1=min(Y_f1');
hyf1=max(Y_f1')-min(Y_f1');
X_train_f1_Normal=(((X_f1'-ones(1000,1)*Minxf1)./(ones(1000,1)*hxf1))*2-1)';
Y_train_f1_Normal=(((Y_f1'-ones(1000,1)*Minyf1)./(ones(1000,1)*hyf1))*2-1)';
% NN for fault1 data 
netf1=newff(X_train_f1_Normal,Y_train_f1_Normal,Nf1,{'tansig' 'purelin'});   
netf1=init(netf1);
netf1.trainFcn = 'trainlm';
netf1.trainParam.epochs = 1;    
for i=1:Epochs_f1
    [netf1,trf1]=train(netf1,X_train_f1_Normal,Y_train_f1_Normal);
    R=randperm(1000);
    X_train_f1_Normal=X_train_f1_Normal(:,R);
    Y_train_f1_Normal=Y_train_f1_Normal(:,R);   
end

%% fault2
% fault2 data
X_f2=[[0,u(1,1:999)];[0,0,y_f2(1,1:998)]];
Y_f2=y_f2;
% normalize
Minxf2=min(X_f2');
hxf2=max(X_f2')-min(X_f2');
Minyf2=min(Y_f2');
hyf2=max(Y_f2')-min(Y_f2');
X_train_f2_Normal=(((X_f2'-ones(1000,1)*Minxf2)./(ones(1000,1)*hxf2))*2-1)';
Y_train_f2_Normal=(((Y_f2'-ones(1000,1)*Minyf2)./(ones(1000,1)*hyf2))*2-1)';
% NN for fault2 data 
netf2=newff(X_train_f2_Normal,Y_train_f2_Normal,Nf2,{'tansig' 'purelin'});   
netf2=init(netf2);
netf2.trainFcn = 'trainlm';
netf2.trainParam.epochs = 1;    
for i=1:Epochs_f2
    [netf2,trf2]=train(netf2,X_train_f2_Normal,Y_train_f2_Normal);
    R=randperm(1000);
    X_train_f2_Normal=X_train_f2_Normal(:,R);
    Y_train_f2_Normal=Y_train_f2_Normal(:,R);   
end

%% Test & Residual
% normal
X_nor=[u_t;[0,u_t(1,1:1299)];[0,0,0,u_t(1,1:1297)];[0,y_t(1,1:1299)];[0,0,y_t(1,1:1298)];[0,0,0,y_t(1,1:1297)]];
Y_main_n=(((y_t'-ones(1300,1)*MinyN)./(ones(1300,1)*hyN))*2-1)';
X_t_n=(((X_nor'-ones(1300,1)*MinxN)./(ones(1300,1)*hxN))*2-1)';
Y_t_n=sim(netN,X_t_n);
Residual_n = Y_main_n - Y_t_n;

% f1
X_fault1=[u_t;[0,u_t(1,1:1299)];[0,0,y_t(1,1:1298)]];
Y_main_f1=(((y_t'-ones(1300,1)*Minyf1)./(ones(1300,1)*hyf1))*2-1)';
X_t_f1=(((X_fault1'-ones(1300,1)*Minxf1)./(ones(1300,1)*hxf1))*2-1)';
Y_t_f1=sim(netf1,X_t_f1);
Residual_f1 = Y_main_f1 - Y_t_f1;

% f2
X_fault2=[[0,u_t(1,1:1299)];[0,0,y_t(1,1:1298)]];
Y_main_f2=(((y_t'-ones(1300,1)*Minyf2)./(ones(1300,1)*hyf2))*2-1)';
X_t_f2=(((X_fault2'-ones(1300,1)*Minxf2)./(ones(1300,1)*hxf2))*2-1)';
Y_t_f2=sim(netf2,X_t_f2);
Residual_f2 = Y_main_f2 - Y_t_f2;

figure
subplot(311)
plot(Residual_n)
xlabel('Time')
ylabel('Residual')
title('Normal Resedual')
grid on
subplot(312)
plot(Residual_f1)
xlabel('Time')
ylabel('Residual')
title('First Fault Resedual')
grid on
subplot(313)
plot(Residual_f2)
xlabel('Time')
ylabel('Residual')
title('Second Fault Resedual')
grid on












