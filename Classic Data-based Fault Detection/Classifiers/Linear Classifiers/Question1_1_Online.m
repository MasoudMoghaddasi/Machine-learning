clc;clear; close all
load('PenBasedRec_15f.mat')
%% Training
W=cell(9,10);
ro=0.01;
for i=1:9
    for k=i+1:10
        Z=[[ones(719,1) Xtrain_new(:,:,i)]; -[ones(719,1) Xtrain_new(:,:,k)]]';
        w=rand(16,1);
        f=1;
        n=2;
        clear h;
        while(f==1 && n<=10000)
            g=w(:,n-1)'*Z;
            h(n-1)=size((g(g<=0)),2);
            w(:,n)=w(:,n-1);
            for j=1:1438
                z=Z(:,j);
                if ((w(:,n)'*z)<=0)
                    w(:,n)=w(:,n)+ro*z;
                end
            end
            if h(n-1)==0
                f=0;
            end
            n=n+1;
        end
        g=w(:,n-1)'*Z;
        h(n-1)=size((g(g<=0)),2);
        W{i,k}=struct('w',w,'h',h);
        i,k
    end
end


for i=1:9
    for k=i+1:10
        [m n]=min(W{i,k}.h);
        w=W{i,k}.w(:,n);
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

save('result1_1_online.mat','W_opt','Confidence')