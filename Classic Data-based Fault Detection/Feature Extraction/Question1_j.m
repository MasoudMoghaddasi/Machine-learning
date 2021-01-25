clc;clear;close all
load HW2_part1.mat

d=input('Please Enter Reducted Dimention:   ');
k=input('Please Enter Number Of Neighbors:   ');
Dyn=input('Please Enter Number Of Dynamics:(for exampel 1 for original data)   ');

X_Train_All={class1,class2,class3};
X_Test_All={test_class1,test_class2,test_class3};


%%
ClassNo=size(X_Train_All,2);
%%
for i=1:ClassNo
    [ X_Train_All{i},X_Test_All{i} ] = Dynamic( X_Train_All{i},X_Test_All{i},Dyn );
end

%%

Z_train=cell(1,ClassNo);
Z_test=cell(1,ClassNo);

%% LDA 
if d<=ClassNo-1
    [ G ] = LDA( X_Train_All , ClassNo , d );
    %%
    for i=1:ClassNo
        Z_train{i}=G*X_Train_All{i}';
        Z_test{i}=G*X_Test_All{i}';
    end

    %%
    if d==2
        figure
        plot(Z_train{1}(1,:),Z_train{1}(2,:),'*')
        hold on
        plot(Z_train{2}(1,:),Z_train{2}(2,:),'r+')
        hold on
        plot(Z_train{3}(1,:),Z_train{3}(2,:),'go')
        hold off
        grid on
    else
        figure
        plot(Z_train{1}(1,:),'*')
        hold on
        plot(Z_train{2}(1,:),'r+')
        hold on
        plot(Z_train{3}(1,:),'go')
        hold off
        grid on
    end
    title('Train Data After LDA')
    if d==2
        figure
        plot(Z_test{1}(1,:),Z_test{1}(2,:),'*')
        hold on
        plot(Z_test{2}(1,:),Z_test{2}(2,:),'r+')
        hold on
        plot(Z_test{3}(1,:),Z_test{3}(2,:),'go')
        hold off
        grid on
    else
        figure
        plot(Z_test{1}(1,:),'*')
        hold on
        plot(Z_test{2}(1,:),'r+')
        hold on
        plot(Z_test{3}(1,:),'go')
        hold off
        grid on
    end
    title('Test Data After LDA')
    %%
    XtrainAll=[];
    Xtest=[];
    ClassVector_train=[];
    ClassNo=size(X_Train_All,2);

    for i=1:ClassNo
        XtrainAll=[XtrainAll ;Z_train{i}'];
        [d ~]=size(Z_train{i}');
        ClassVector_train=[ClassVector_train; i*ones(d,1)];
        Xtest=[Xtest ;Z_test{i}'];
        [f(i) ~]=size(Z_test{i}');
    end
    %%

    Class_Out=KNN_classifier(XtrainAll',Xtest',ClassVector_train,k);
    %%
    for n=1:ClassNo
        C=Class_Out(1:f(n));
        Class_Out(1:f(n))=[];
        for j=1:ClassNo+1
           Confidence(n,j)=size(C(C==j),2)/f(n);
        end
    end
    Confidence

    save('result2_4_KNN','Confidence')
else
    disp('Your requested dimention is not valid')
end