clc;clear all
load('PenBasedRec_15f.mat')

ClassVector_train=[];
Xvalidation=[];
Xtest=[];
XtrainAll=[];
H=1:5:101;

for p=1:100
    n=ceil(size(Xtrain_new,1)*rand);
    m=ceil(size(Xtest_new,1)*rand);
    for q=1:10
        Xvalidation1{p,q}=[Xtrain_new(n,:,q);Xtest_new(m,:,q)];
    end
    Xtrain_new(n,:,:)=[];
    Xtest_new(m,:,:)=[];
end
d=size(Xtrain_new,1);
f=size(Xtest_new,1);
y=200;
for i=1:10
    XtrainAll=[XtrainAll ;Xtrain_new(:,:,i)];
    for h=1:100
        Xvalidation=[Xvalidation;Xvalidation1{h,i}];
    end
    ClassVector_train=[ClassVector_train; i*ones(d,1)];
    Xtest=[Xtest ;Xtest_new(:,:,i)];
end
Confidence=zeros(10,11,size(H,2));
for h=1:size(H,2)
    Class_Out(h,:)=GaussKer_classifier(XtrainAll',Xtest',ClassVector_train,H(h));
    for n=1:10
        C=Class_Out(h,(n-1)*f+1:n*f);
        for j=1:11
           Confidence(n,j,h)=size(C(C==j),2)/f;
        end
    end
    TRACE(h)=trace(Confidence(:,1:end-1,h));
end

plot(H,TRACE)
title('Trace of confidence matrices Vs. "h"')
xlabel('h')
ylabel('Trace')

[~,h_opt]=max(TRACE);
H_opt=H(h_opt)

class_Out=GaussKer_classifier(XtrainAll',Xvalidation',ClassVector_train,H_opt);
confidence=zeros(10,11);
for n=1:10
    C=class_Out((n-1)*y+1:n*y);
    for j=1:11
       confidence(n,j)=size(C(C==j),2)/y;
    end
end
confidence
save('result2_3_gauss','H_opt','confidence')