clc;clear all;close all
load('PenBasedRec_15f.mat')
mu=mean(Xtrain_new(:,:,1:2));
N=size(Xtrain_new(:,:,1:2),1);
f=size(Xtest_new(:,:,1:2),1);
sigma=zeros(15,15,2);
for i=1:2
    z=Xtrain_new(:,:,i)-(ones(N,1)*mu(:,:,i));
    sigma(:,:,i)=(1/(N-1))*(z'*z);
end

Class_Out=zeros(f,2);
Landa=[0.1 1;0.100000001 0.1];
for i=1:2
    for k=1:f
        Class_Out(k,i)=gaussianpdf_risk_classifier(Xtest_new(k,:,i),mu,sigma,Landa);
    end
end
        

Confidence=zeros(2,3);
for n=1:2
    C=Class_Out(:,n);
    for j=1:3
       Confidence(n,j)=size(C(C==j),1)/f;
    end
end
Confidence

save('result3_2_2.mat','Confidence','Landa');