clc;clear; close all
load('Data.mat')
load('Unknown.mat')
%% Training
W=cell(2,3);
ro=0.01;
Xtrain_new(:,:,1)=[normal(1:200,:) normal(2:201,:)];
Xtrain_new(:,:,2)=[f1(1:200,:) f1(2:201,:)];
Xtrain_new(:,:,3)=[f2(1:200,:) f2(2:201,:)];

Xtest_new(:,:,1)=[test1(1:200,:) test1(2:201,:)];
Xtest_new(:,:,2)=[test2(1:200,:) test2(2:201,:)];
Xtest_new(:,:,3)=[test3(1:200,:) test3(2:201,:)];

st=size(Xtrain_new);
ss=size(Xtest_new);
for i=1:st(3)-1
    for k=i+1:st(3)
        b=rand(2*st(1),1);
        z=[[ones(st(1),1) Xtrain_new(:,:,i)]; -[ones(st(1),1) Xtrain_new(:,:,k)]];
        w=(z'*z)^-1*z'*b; 
        g=w'*z';
        h=size((g(g<=0)),2);
        W_opt{i,k}=struct('w',w,'h',h);
    end
end



%%
Confidence=zeros(st(3),st(3)+1);
for n=1:ss(3)
    for j=1:ss(1)
        z=[1;Xtest_new(j,:,n)'];
        for i=1:st(3)-1
            for k=i+1:st(3)
                if (W_opt{i,k}.w'*z)>0
                    y(i,k)=i;
                else
                    y(i,k)=k;
                end
            end
        end
        Class=st(3)+1;
        m=0;
        for c=1:st(3)
            [u ~]=size(y(y==c));
            if u>m
                Class=c;
                m=u;
            else if (u==m && u~=0)
                    Class=st(3)+1;
                end
            end
        end
        Confidence(n,Class)=Confidence(n,Class)+1;
    end
end
Confidence=Confidence/ss(1)

save('result4_2.mat','W_opt','Confidence','b')