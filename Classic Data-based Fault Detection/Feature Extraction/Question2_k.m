clc;clear;close all
load HW2_part1.mat

d=input('Please Enter Reducted Dimention:   ');
X_Train_All={class1,class2,class3};
X_Test_All={test_class1,test_class2,test_class3};


%%
ClassNo=size(X_Train_All,2);

%% Whitenning
[Gw , mu] = Whitening( X_Train_All,ClassNo );

for i=1:ClassNo
    X_Train_All{i}=(X_Train_All{i}-ones(size(X_Train_All{i},1),1)*mu)*Gw';
    X_Test_All{i}=(X_Test_All{i}-ones(size(X_Test_All{i},1),1)*mu)*Gw';
end
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

if d==3
    figure
    plot3(Z_train{1}(1,:),Z_train{1}(2,:),Z_train{1}(3,:),'*')
    hold on
    plot3(Z_train{2}(1,:),Z_train{2}(2,:),Z_train{2}(3,:),'r+')
    hold on
    plot3(Z_train{3}(1,:),Z_train{3}(2,:),Z_train{3}(3,:),'go')
    hold off
    grid on
legend('First Class','Second Class','Third Class');
title('Train Features After PCA')
elseif d==2
    figure
    plot(Z_train{1}(1,:),Z_train{1}(2,:),'*')
    hold on
    plot(Z_train{2}(1,:),Z_train{2}(2,:),'r+')
    hold on
    plot(Z_train{3}(1,:),Z_train{3}(2,:),'go')
    hold off
    grid on
legend('First Class','Second Class','Third Class');
title('Train Features After PCA')
elseif d==1
    figure
    plot(Z_train{1}(1,:),'*')
    hold on
    plot(Z_train{2}(1,:),'r+')
    hold on
    plot(Z_train{3}(1,:),'go')
    hold off
    grid on
legend('First Class','Second Class','Third Class');
title('Train Features After PCA')
end

if d==3
    figure
    plot3(Z_test{1}(1,:),Z_test{1}(2,:),Z_test{1}(3,:),'*')
    hold on
    plot3(Z_test{2}(1,:),Z_test{2}(2,:),Z_test{2}(3,:),'r+')
    hold on
    plot3(Z_test{3}(1,:),Z_test{3}(2,:),Z_test{3}(3,:),'go')
    hold off
    grid on
    legend('First Class','Second Class','Third Class');
    title('Test Features After PCA')
elseif d==2
    figure
    plot(Z_test{1}(1,:),Z_test{1}(2,:),'*')
    hold on
    plot(Z_test{2}(1,:),Z_test{2}(2,:),'r+')
    hold on
    plot(Z_test{3}(1,:),Z_test{3}(2,:),'go')
    hold off
    grid on
    legend('First Class','Second Class','Third Class');
    title('Test Features After PCA')
elseif d==1
    figure
    plot(Z_test{1}(1,:),'*')
    hold on
    plot(Z_test{2}(1,:),'r+')
    hold on
    plot(Z_test{3}(1,:),'go')
    hold off
    grid on
    legend('First Class','Second Class','Third Class');
    title('Test Features After PCA')
end

W_opt = Classifier_LS( Z_train , ClassNo );
% W_opt = Classifier_Batch( Z_train , ClassNo , 0.01 , 2000 );


%%
confidence=Confidence( W_opt,Z_test,ClassNo )


%%
save('result1_a.mat','W_opt','confidence','d')