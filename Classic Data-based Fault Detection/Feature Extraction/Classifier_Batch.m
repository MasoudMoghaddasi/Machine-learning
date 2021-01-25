function [ W_opt ] = Classifier_Batch( Xtrain_new , ClassNo , ro , epoch )
W=cell(ClassNo-1,ClassNo);
for i=1:ClassNo-1
    for k=i+1:ClassNo
        [m1, n1]=size(Xtrain_new{i}');
        [m2, ~]=size(Xtrain_new{k}');
        w=rand(n1+1,1);
        f=1;
        n=2;
        clear h;
        while(f==1 && n<=epoch)
            h(n-1)=0;
            Y=[];
            for j=1:m1
                z=[1;Xtrain_new{i}(:,j)];
                if ((w(:,n-1)'*z)<=0)
                    h(n-1)=h(n-1)+1;
                    Y=[Y z];
                end
            end
            for j=1:m2
                z=-[1;Xtrain_new{k}(:,j)];
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
        for j=1:m1
            z=[1;Xtrain_new{i}(:,j)];
            if ((w(:,n-1)'*z)<=0)
                h(n-1)=h(n-1)+1;
            end
        end
        for j=1:m2
            z=-[1;Xtrain_new{k}(:,j)];
            if ((w(:,n-1)'*z)<=0)
                h(n-1)=h(n-1)+1;
            end
        end
        W{i,k}=struct('w',w,'h',h);
    end
end


for i=1:ClassNo-1
    for k=i+1:ClassNo
        [m n]=min(W{i,k}.h);
        w=W{i,k}.w(:,n);
        W_opt{i,k}=struct('w',w,'h',m);
    end
end

end

