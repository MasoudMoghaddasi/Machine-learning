function Class_Out=GaussKer_classifier(XtrainAll,Xtest,ClassVector_train,h)
    [~,N1]=size(XtrainAll);
    [~,N]=size(Xtest);
    c=max(ClassVector_train); 
    for i=1:N
        Sum=zeros(1,c);
        for j=1:N1
            fi=exp(-((XtrainAll(:,j)-Xtest(:,i))'*(XtrainAll(:,j)-Xtest(:,i)))/(2*h^2));
            Sum(1,ClassVector_train(j))=Sum(1,ClassVector_train(j))+fi;
        end
        [MAX,Class_Out(i)]=max(Sum);
        Sum(Class_Out(i))=[];
        if Sum~=MAX;
        else
            Class_Out(i)=c+1;
        end
    end
end