clc;clear; close all
load('PenBasedRec_15f.mat')
%% Training
W=cell(9,10);
ro=0.01;
for i=1:9
    for k=i+1:10
        w=rand(16,1);
        f=1;
        n=2;
        clear h;
        while(f==1 && n<=10000)
            h(n-1)=0;
            Y=[];
            for j=1:719
                z=[1;Xtrain_new(j,:,i)'];
                if ((w(:,n-1)'*z)<=0)
                    h(n-1)=h(n-1)+1;
                    Y=[Y z];
                end
                z=-[1;Xtrain_new(j,:,k)'];
                if ((w(:,n-1)'*z)<=0)
                    h(n-1)=h(n-1)+1;
                    Y=[Y z];
                end
            end
            if (~isempty(Y))
                w(:,n)=w(:,n-1)+ro*sum(Y,2);
            else
               w(:,n)=w(:,n-1);
            end
            if h(n-1)==0
                f=0;
            end
            n=n+1;
        end
        h(n-1)=0;
        for j=1:719
            z=[1;Xtrain_new(j,:,i)'];
            if ((w(:,n-1)'*z)<=0)
                h(n-1)=h(n-1)+1;
            end
            z=-[1;Xtrain_new(j,:,k)'];
            if ((w(:,n-1)'*z)<=0)
                h(n-1)=h(n-1)+1;
            end
        end
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

save('result1_1_offline.mat','W_opt','Confidence')