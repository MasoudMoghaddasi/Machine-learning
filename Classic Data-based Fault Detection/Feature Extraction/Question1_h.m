clc;clear;close all
load HW2_part1.mat


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

    for i=1:ClassNo-1
        [ G{i} , E(i) ] = LDA( X_Train_All , ClassNo , i );
    end
    disp('Fisher Criteria:')
    disp(E)
    d=input('Please Enter Reducted Dimention:   ');
if d<=ClassNo-1
    %%
    for i=1:ClassNo
        Z_train{i}=G{d}*X_Train_All{i}';
        Z_test{i}=G{d}*X_Test_All{i}';

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
    title('Train Features After PCA')
    elseif d==1
        figure
        plot(Z_train{1}(1,:),'*')
        hold on
        plot(Z_train{2}(1,:),'r+')
        hold on
        plot(Z_train{3}(1,:),'go')
        hold off
        grid on
    legend('First Class','Second Class','Third Class');
    title('Train Features After PCA')
    end

    %%
    W_opt = Classifier_LS( Z_train , ClassNo );
    % W_opt = Classifier_Batch( Z_train , ClassNo , 0.01 , 2000 );


    %%
    confidence=Confidence( W_opt,Z_test,ClassNo )


    %%
    save('result1_a.mat','W_opt','confidence','d')

else
    disp('Your requested dimention is not valid')
end
