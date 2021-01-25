function [ G , mu ] = Whitening( X_Train_All,ClassNo )

    x=[];
    n=zeros(1,ClassNo);
    for i=1:ClassNo
        x=[x; X_Train_All{i}];
        [n(i) ~]=size(X_Train_All{i});
    end
    mu=mean(x);
    X=x-ones(sum(n),1)*mu;
    S=(X'*X)/sum(n);
    [V D]=eig(S);
    G=D^(-1/2)*V';
    Z=G*X';

end

