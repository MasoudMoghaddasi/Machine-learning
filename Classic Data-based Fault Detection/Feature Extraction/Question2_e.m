clc;clear;close all
load HW2_part1.mat

d=input('Please Enter Reducted Dimention:   ');
Dyn=input('Please Enter Number Of Dynamics:(for exampel 1 for original data)   ');

X_Train_All={class1,class2,class3};
X_Test_All={test_class1,test_class2,test_class3};

%%
ClassNo=size(X_Train_All,2);

Z_train=cell(1,ClassNo);
Z_test=cell(1,ClassNo);
%% Whitenning
[Gw , mu] = Whitening( X_Train_All,ClassNo );

for i=1:ClassNo
    X_Train_All{i}=(X_Train_All{i}-ones(size(X_Train_All{i},1),1)*mu)*Gw';
    X_Test_All{i}=(X_Test_All{i}-ones(size(X_Test_All{i},1),1)*mu)*Gw';
end
%% PCA
[  G , E ] = PCA( X_Train_All , d ,ClassNo );

%%

for i=1:ClassNo
    Z_Tr=G*X_Train_All{i}';
    Z_Te=G*X_Test_All{i}';
    [ ZTRAIN,ZTEST ] = Dynamic( Z_Tr',Z_Te',Dyn );
    Z_train{i}=ZTRAIN';
    Z_test{i}=ZTEST';
end

%%
W_opt = Classifier_LS( Z_train , ClassNo );
% W_opt = Classifier_Batch( Z_train , ClassNo , 0.01 , 2000 );


%%
confidence=Confidence( W_opt,Z_test,ClassNo )


%%
% save('result1_a.mat','W_opt','confidence','d')
