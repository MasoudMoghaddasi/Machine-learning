clc;clear; close all
load('PenBasedRec_15f.mat')
%% Training
W=cell(9,10);
ro=0.0000002;
for i=1:9
    for k=i+1:10
        b=rand(1438,1);
        Z=[[ones(719,1) Xtrain_new(:,:,i)]; -[ones(719,1) Xtrain_new(:,:,k)]]';
        w=rand(16,1);
        f=1;
        n=2;
        clear h;
        while(f==1 && n<=1000)
            g=w(:,n-1)'*Z;
            h(n-1)=size((g(g<=0)),2);
            w(:,n)=w(:,n-1);
            for j=1:1438
                z=Z(:,j);
                w(:,n)=w(:,n)+ro*z*(b(j)-z'*w(:,n));
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

plot(W{1,9}.h)
title('no. of isclasses Vs. iteration for w_1_9')
xlabel('iteration')
ylabel('no. of misclasses')
figure
plot(W{5,7}.h)
title('no. of isclasses Vs. iteration for w_5_7')
xlabel('iteration')
ylabel('no. of misclasses')
figure
plot(W{8,9}.h)
title('no. of isclasses Vs. iteration for w_8_9')
xlabel('iteration')
ylabel('no. of misclasses')

save('result1_3.mat','W_opt','Confidence')