clc;clear all
load('PenBasedRec_15f.mat')

ClassVector_train=[];
Xvalidation=[];
Xtest=[];
XtrainAll=[];

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

for k=1:100
    Class_Out(k,:)=KNN_classifier(XtrainAll',Xtest',ClassVector_train,k);
    Confidence=zeros(10,11,100);
    for n=1:10
        C=Class_Out(k,(n-1)*f+1:n*f);
        for j=1:11
           Confidence(n,j,k)=size(C(C==j),2)/f;
        end
    end
    TRACE(k)=trace(Confidence(:,1:10,k));
end

plot(TRACE)
title('Trace of confidence matrices Vs. "K"')
xlabel('K')
ylabel('Trace')

[~,K_opt]=max(TRACE)

class_Out=KNN_classifier(XtrainAll',Xvalidation',ClassVector_train,K_opt);
confidence=zeros(10,11);
for n=1:10
    C=class_Out((n-1)*y+1:n*y);
    for j=1:11
       confidence(n,j)=size(C(C==j),2)/y;
    end
end
confidence
save('result2_3_KNN','K_opt','confidence')