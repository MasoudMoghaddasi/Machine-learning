clc;clear; close all
load('PenBasedRec_15f.mat','Xtrain_new')
%% Training
W=cell(9,10,10);
ro=0.01;
for t=1:10
    for i=1:9
        for k=i+1:10
            w=rand(16,1);
            f=1;
            n=2;
            clear h;
            if(t==1)
                C=1;
            else
                C=min(W{i,k,t-1}.h);
            end
            if(C~=0)
                %%
                while(f==1 && n<=15000)
                    %% h(n) Shows Misclass Datas
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
                    %%
                    w(:,n)=w(:,n-1);
                    for j=1:719
                        z=[1;Xtrain_new(j,:,i)'];
                        if ((w(:,n)'*z)<=0)
                            w(:,n)=w(:,n)+ro*z;
                        end
                    end
                    for j=1:719
                        z=-[1;Xtrain_new(j,:,k)'];
                        if ((w(:,n)'*z)<=0)
                            w(:,n)=w(:,n)+ro*z;
                        end
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
                %%
            else
                h=0;
                w=W{i,k,t-1}.w(:,end);
            end
            i
            k
            W{i,k,t}=struct('w',w,'h',h);
        end
    end
end


for i=1:9
    for k=i+1:10
        for t=1:10
            [m(t) n]=min(W{i,k,t}.h);
            w_opt(t)=W{i,k,t}.w(:,n);
        end
        [u v]=min(m(t));
        W_opt{i,k}=struct('w',w_opt(v),'h',u);
    end
end


save('result.mat','W','W_opt')
