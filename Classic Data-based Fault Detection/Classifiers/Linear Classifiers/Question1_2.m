clc;clear; close all
load('PenBasedRec_15f.mat')
%% Training
W=cell(9,10);
ro=0.01;
for i=1:9
    for k=i+1:10
        b=rand(1438,1);
        z=[[ones(719,1) Xtrain_new(:,:,i)]; -[ones(719,1) Xtrain_new(:,:,k)]];
        w=(z'*z)^-1*z'*b; 
        g=w'*z';
        h=size((g(g<=0)),2);
        W_opt{i,k}=struct('w',w,'h',h);
    end
end

%%
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

save('result1_2_b2.mat','W_opt','Confidence','b')