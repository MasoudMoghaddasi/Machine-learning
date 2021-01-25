clc;clear;close all
load HW2_part1.mat

d=input('Please Enter Reducted Dimention:   ');
k=input('Please Enter Number Of Neighbors:   ');
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
XtrainAll=[];
Xtest=[];
ClassVector_train=[];
ClassNo=size(X_Train_All,2);

for i=1:ClassNo
    XtrainAll=[XtrainAll ;Z_train{i}'];
    [d ~]=size(Z_train{i}');
    ClassVector_train=[ClassVector_train; i*ones(d,1)];
    Xtest=[Xtest ;Z_test{i}'];
    [f(i) ~]=size(Z_test{i}');
end
%%

Class_Out=KNN_classifier(XtrainAll',Xtest',ClassVector_train,k);
%%
for n=1:ClassNo
    C=Class_Out(1:f(n));
    Class_Out(1:f(n))=[];
    for j=1:ClassNo+1
       Confidence(n,j)=size(C(C==j),2)/f(n);
    end
end
Confidence

% save('result2_4_KNN','Confidence')