clc;clear; close all
load('Data.mat')
load('Unknown.mat')
%% Training
W=cell(2,3);
ro=0.01;
Xtrain_new(:,:,1)=normal;
Xtrain_new(:,:,2)=f1;
Xtrain_new(:,:,3)=f2;

Xtest_new(:,:,1)=test4;

st=size(Xtrain_new);
ss=size(Xtest_new);
if size(ss,2)==2
    ss(3)=1;
end
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
Class=zeros(ss(3),ss(1));
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
        m=0;
        for c=1:st(3)
            [u ~]=size(y(y==c));
            if u>m
                Class(n,j)=c;
                m=u;
            else if (u==m && u~=0)
                    Class(n,j)=st(3)+1;
                end
            end
        end
    end
end
j=1;
while Class(j)==1
    fault_time=tout(j+1);
    fault_kind=Class(n,j+1);
    j=j+1;
end
fault_time
fault_kind
save('result4_3.mat','W_opt','Class','fault_time','fault_kind')