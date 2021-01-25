clc;clear; close all
load('PenBasedRec_15f.mat')
%% Training
W=cell(9,10);
ro=0.00000015;
for i=1:9
    for k=i+1:10
        b=rand(1438,1);
        Z=[[ones(719,1) Xtrain_new(:,:,i)]; -[ones(719,1) Xtrain_new(:,:,k)]]';
        w=rand(16,1);
        g=w(:,1)'*Z;
        h(1)=size((g(g<=0)),2);
        w(:,2)=w(:,1);
        for j=1:1438
            z=Z(:,j);
            w(:,2)=w(:,2)+ro*z*(b(j)-z'*w(:,2));
        end
        g=w(:,2)'*Z;
        h(2)=size((g(g<=0)),2);
        W{i,k}=struct('w',w,'h',h);
    end
end

for i=1:9
    for k=i+1:10
        m=W{i,k}.h(2);
        w=W{i,k}.w(:,2);
        W_opt{i,k}=struct('w',w,'h',m);
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



save('result1_3_oneloop.mat','W_opt','Confidence')