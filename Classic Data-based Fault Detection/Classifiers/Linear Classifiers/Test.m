clc;clear;close all
load('result.mat','W_opt')
load('PenBasedRec_15f.mat','Xtest_new');
Confidence=zeros(10,11);
for n=1:10
    for j=1:335
        z=[1;Xtest_new(j,:,n)'];
        for i=1:9
            for k=i+1:10
                if (W_opt{i,k}.w'*z)>0
                    y(i,k)=i;
                else
                    y(i,k)=k;
                end
            end
        end
        Class=11;
        m=0;
        for c=1:10
            [u ~]=size(y(y==c));
            if u>m
                Class=c;
                m=u;
            else if (u==m && u~=0)
                    Class=11;
                end
            end
        end
        Confidence(n,Class)=Confidence(n,Class)+1;
    end
end
Confidence=Confidence/335