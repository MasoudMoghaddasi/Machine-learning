clc;clear all;close all
load('PenBasedRec_15f.mat')
mu=mean(Xtrain_new);
N=size(Xtrain_new,1);
f=size(Xtest_new,1);
sigma=zeros(15,15,10);
for i=1:10
    z=Xtrain_new(:,:,i)-(ones(N,1)*mu(:,:,i));
    sigma(:,:,i)=(1/(N-1))*(z'*z);
end

Class_Out=zeros(f,10);
for i=1:10
    for k=1:f
        Class_Out(k,i)=gaussianpdf_classifier(Xtest_new(k,:,i),mu,sigma);
    end
end
        

Confidence=zeros(10,11);
for n=1:10
    C=Class_Out(:,n);
    for j=1:11
       Confidence(n,j)=size(C(C==j),1)/f;
    end
end
Confidence

save('result3_1.mat','Confidence');