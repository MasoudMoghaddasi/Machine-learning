clc;clear;close all
load HW2_part1.mat

d=input('Please Enter Reducted Dimention:   ');
Nern_Max=input('Please Enter Maximum Number Of Neurons:   ');
MsE=input('Please Enter MSE For RBF:   ');
X_Train_All={class1,class2,class3};
X_Test_All={test_class1,test_class2,test_class3};


%%
ClassNo=size(X_Train_All,2);
xtr=[];
for i=1:ClassNo
    xtr=[xtr; X_Train_All{i}];
end


options.KernelType = 'Gaussian';
options.t = 1;
options.ReducedDim = d;
[eigvector,eigvalue] = KPCA(xtr,options);


Z_train=cell(1,ClassNo);
Z_test=cell(1,ClassNo);
%%
for i=1:ClassNo
    Ktrain = constructKernel(X_Train_All{i},xtr,options);
    Z_train{i} = (Ktrain*eigvector)';
    Ktest = constructKernel(X_Test_All{i},xtr,options);
    Z_test{i} = (Ktest*eigvector)';
end
%%
%%

X_Train=X_Train_All;
for i=1:ClassNo
    [n(i) ~]=size(X_Train_All{i});
    sel{i}=randperm(n(i));
    X_Test{i}=X_Train_All{i}(sort(sel{i}(1:floor(0.25*n(i)))),:);
    X_Train{i}(sort(sel{i}(1:floor(0.25*n(i)))),:)=[];
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
    y{i}(sort(sel{i}(1:floor(0.25*n(i)))),:)=[];
    Y=[Y;y{i}];
end

%%
net=newrb(X_train_Normal',Y',MsE,1,Nern_Max,5);   
net=init(net);

Out_TR = sim(net,X_train_Normal');
E_TR=Out_TR-Y';
Out_TE = sim(net,X_test_Normal');
E_TE=Out_TE-Y_test';
MSE_TR=mse(E_TR)
MSE_TE=mse(E_TE)


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

Out_TR = sim(net,X_train_Normal');
Out_TE = sim(net,X_test_Normal');
fig=1;
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
