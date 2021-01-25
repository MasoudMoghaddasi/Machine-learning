clc;clear;close all
load HW2_part1.mat

d=input('Please Enter Reducted Dimention:   ');
Dyn=input('Please Enter Number Of Dynamics:(for exampel 1 for original data)   ');
Nern_Num_Lim=input('Please Enter Minimum And Maximum Number Of Neurons:[min max]   ');
Nern_Step=input('Please Enter Step For Encreasing Neurons:   ');
Epochs =input('Please Enter Number Of Epochs:   ');

X_Train_All={class1,class2,class3};
X_Test_All={test_class1,test_class2,test_class3};


%%
ClassNo=size(X_Train_All,2);
%%
%% Whitenning
[Gw , mu] = Whitening( X_Train_All,ClassNo );

for i=1:ClassNo
    X_Train_All{i}=(X_Train_All{i}-ones(size(X_Train_All{i},1),1)*mu)*Gw';
    X_Test_All{i}=(X_Test_All{i}-ones(size(X_Test_All{i},1),1)*mu)*Gw';
end
%%
for i=1:ClassNo
    [ X_Train_All{i},X_Test_All{i} ] = Dynamic( X_Train_All{i},X_Test_All{i},Dyn );
end

%%

Z_train=cell(1,ClassNo);
Z_test=cell(1,ClassNo);

%% LDA 
if d<=ClassNo-1
    [ G ] = LDA( X_Train_All , ClassNo , d );
    %%
    for i=1:ClassNo
        X_Train_All{i}=X_Train_All{i}*G';
        X_Test_All{i}=X_Test_All{i}*G';

    end

    %%
    ClassNo=size(X_Train_All,2);
    X_Train=X_Train_All;
    for i=1:ClassNo
        [n(i) ~]=size(X_Train_All{i});
        sel{i}=randperm(n(i));
        X_Test{i}=X_Train_All{i}(sort(sel{i}(1:floor(0.25*n(i)))),:);
        X_Train{i}(sel{i}(1:floor(0.25*n(i))),:)=[];
    end


    %%


    x_train=[];
    n_train=zeros(1,ClassNo);
    for i=1:ClassNo
        x_train=[x_train; X_Train{i}];
        [n_train(i) n_featur]=size(X_Train{i});
    end

    x_test=[];
    x_test_all=[];
    n_test=zeros(1,ClassNo);
    for i=1:ClassNo
        x_test=[x_test; X_Test{i}];
        [n_test(i) ~]=size(X_Test{i});
        x_test_all=[x_test_all;X_Test_All{i}];
        [f(i) ~]=size(X_Test_All{i});
    end

    Min=min(x_train);
    h=max(x_train)-min(x_train);

    %%
    X_train_Normal=((x_train-ones(size(x_train,1),1)*Min)./(ones(size(x_train,1),1)*h))*2-1;
    X_test_Normal=((x_test-ones(size(x_test,1),1)*Min)./(ones(size(x_test,1),1)*h))*2-1;
    x_test_all_Normal=((x_test_all-ones(size(x_test_all,1),1)*Min)./(ones(size(x_test_all,1),1)*h))*2-1;
    %% 

    for i=1:ClassNo
        a=-1+(i-1)*(2/ClassNo);
        m=2/ClassNo;
        y{i}=m*rand(size(X_Train_All{i},1),1)+a;
    end

    Y_test=[];
    Y=[];
    for i=1:ClassNo
        Y_test=[Y_test;y{i}(sort(sel{i}(1:floor(0.25*n(i)))),:)];
        y{i}(sel{i}(1:floor(0.25*n(i))),:)=[];
        Y=[Y;y{i}];
    end

    %%
    count=0;
    for KN=Nern_Num_Lim(1):Nern_Step:Nern_Num_Lim(2)  
        count=count+1;  
        net=newff(X_train_Normal',Y',KN,{'tansig' 'purelin'});   
        net=init(net);
        net.trainFcn = 'trainlm';
        net.trainParam.epochs = 1;
           
        SSE_TR=[];
        SSE_TE=[];
        for i=1:Epochs
            [net,tr]=train(net,X_train_Normal',Y');
            SSE_TR_a = sum( (Y'-sim(net,X_train_Normal')).^2 );
            SSE_TE_a = sum( (Y_test'-sim(net,X_test_Normal')).^2 );
            SSE_TR = [SSE_TR SSE_TR_a];
            SSE_TE  =[SSE_TE SSE_TE_a];
            Rshfle=randperm(sum(n_train));
            X_train_Normal=X_train_Normal(Rshfle,:);
            Y=Y(Rshfle,:);   
        end
        Out_TR = sim(net,X_train_Normal');
        E_TR=Out_TR-Y';
        MSE_TR(count)=mse(E_TR); 
        Out_TE = sim(net,X_test_Normal');
        E_TE=Out_TE-Y_test';
        MSE_TE(count)=mse(E_TE);
    end


    %% Confidence
    y = sim(net,x_test_all_Normal');
    Class_Out=4*ones(size(y));
    
    
    for i=1:ClassNo
        a=-1+(i-1)*(2/ClassNo);
        m=2/ClassNo;
        Class_Out(y>=a & y <= (m+a))=i;
    end

    for n=1:ClassNo
        C=Class_Out(1:f(n));
        Class_Out(1:f(n))=[];
        for j=1:ClassNo+1
           Confidence(n,j)=size(C(C==j),2)/f(n);
        end
    end
    Confidence
%%
    fig=1;
    figure(fig)
    plot((Nern_Num_Lim(1):Nern_Step:Nern_Num_Lim(2)),MSE_TE(1,:,end),'r')
    hold on
    plot((Nern_Num_Lim(1):Nern_Step:Nern_Num_Lim(2)),MSE_TR(1,:,end))
    set(findall(figure(fig),'type','line'),'linewidth',1)
    grid on
    xlabel('Neuron')
    ylabel('MSE')
    title('MSE Per Neuron')
    legend ('Test','Train')
    fig=fig+1;

    figure(fig)
    plot(1:Epochs,SSE_TE/sum(n_test),'r')
    hold on
    plot(1:Epochs,SSE_TR/sum(n_train))
    set(findall(figure(fig),'type','line'),'linewidth',1)
    grid on
    xlabel('Epochs')
    ylabel('MSE')
    title('MSE Per Epochs For Optimal Number Of Neuron')
    legend ('Test','Train')
    fig=fig+1;


    Out_TR = sim(net,X_train_Normal');
    Out_TE = sim(net,X_test_Normal');

    figure(fig)
    plot(Y)
    hold on
    plot(Out_TR,'r--')
    set(findall(figure(fig),'type','line'),'linewidth',1)
    grid on
    title('y & y_h_a_t For Train')
    legend ('y','y_h_a_t')
    fig=fig+1;


    figure(fig)
    plot(Y_test)
    hold on
    plot(Out_TE,'r--')
    set(findall(figure(fig),'type','line'),'linewidth',1)
    grid on
    title('y & y_h_a_t For Test')
    legend ('y','y_h_a_t')



%     save('result4_4_1','Confidence','net')
else
    disp('Your requested dimention is not valid')
end