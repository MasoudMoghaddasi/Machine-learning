clc;clear;close all
load HW2_part1.mat

d=input('Please Enter Reducted Dimention:   ');

X_Train_All={class1,class2,class3};
X_Test_All={test_class1,test_class2,test_class3};



%%
ClassNo=size(X_Train_All,2);


Z_train=cell(1,ClassNo);
Z_test=cell(1,ClassNo);

%% LDA 
if d<=ClassNo-1
    [ G , E ] = LDA( X_Train_All , ClassNo , d );


    %%
    disp('Projection Matrix=')
    disp(G')
    disp('Fisher Critria=')
    disp(E)
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
        legend('First Class','Second Class','Third Class');
    else
        figure
        plot(Z_train{1}(1,:),'*')
        hold on
        plot(Z_train{2}(1,:),'r+')
        hold on
        plot(Z_train{3}(1,:),'go')
        hold off
        grid on
        legend('First Class','Second Class','Third Class');
    end
    title('Train Features After LDA')
    if d==2
        figure
        plot(Z_test{1}(1,:),Z_test{1}(2,:),'*')
        hold on
        plot(Z_test{2}(1,:),Z_test{2}(2,:),'r+')
        hold on
        plot(Z_test{3}(1,:),Z_test{3}(2,:),'go')
        hold off
        grid on
        legend('First Class','Second Class','Third Class');
    else
        figure
        plot(Z_test{1}(1,:),'*')
        hold on
        plot(Z_test{2}(1,:),'r+')
        hold on
        plot(Z_test{3}(1,:),'go')
        hold off
        grid on
        legend('First Class','Second Class','Third Class');
    end
    title('Test Features After LDA')
    %%
    W_opt = Classifier_LS( Z_train , ClassNo );


    %%
    Confidence=Confidence( W_opt,Z_test,ClassNo )

    %%
    save('result1_b.mat','W_opt','Confidence','d')
else
    disp('Your requested dimention is not valid')
end
