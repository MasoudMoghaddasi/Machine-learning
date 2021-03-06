clc;clear;close all
load HW2_part1.mat

d=input('Please Enter Reducted Dimention:   ');
Dyn=input('Please Enter Number Of Dynamics:(for exampel 1 for original data)   ');
sp=input('Please Enter Spread For PNN:   ');

X_Train_All={class1,class2,class3};
X_Test_All={test_class1,test_class2,test_class3};


%%
ClassNo=size(X_Train_All,2);

%%
for i=1:ClassNo
    [ X_Train_All{i},X_Test_All{i} ] = Dynamic( X_Train_All{i},X_Test_All{i},Dyn );
end

%%

Z_train=cell(1,ClassNo);
Z_test=cell(1,ClassNo);

%% PCA 

[  G , E ] = PCA( X_Train_All , d ,ClassNo );



%%
for i=1:ClassNo
    X_Train_All{i}=X_Train_All{i}*G';
    X_Test_All{i}=X_Test_All{i}*G';
    
end

%%
XtrainAll=[];
Xtest=[];
ClassVector_train=[];
ClassNo=size(X_Train_All,2);

for i=1:ClassNo
    XtrainAll=[XtrainAll ;X_Train_All{i}];
    [d ~]=size(X_Train_All{i});
    ClassVector_train=[ClassVector_train; i*ones(d,1)];
    Xtest=[Xtest ;X_Test_All{i}];
    [f(i) ~]=size(X_Test_All{i});
end

T = ind2vec(ClassVector_train);
net = newpnn(XtrainAll',T,sp);
Y = sim(net,Xtest');
Class_Out=vec2ind(Y);
%%
for n=1:ClassNo
    C=Class_Out(1:f(n));
    Class_Out(1:f(n))=[];
    for j=1:ClassNo+1
       Confidence(n,j)=size(C(C==j),2)/f(n);
    end
end
Confidence

save('result4_4_2','Confidence')