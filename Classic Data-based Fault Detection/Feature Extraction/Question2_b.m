clc;clear;close all
load HW2_part1.mat



X_Train_All={class1,class2,class3};
X_Test_All={test_class1,test_class2,test_class3};
%%
ClassNo=size(X_Train_All,2);
Z_train=cell(1,ClassNo);
Z_test=cell(1,ClassNo);
%% Whitenning
[Gw , mu] = Whitening( X_Train_All,ClassNo );

for i=1:ClassNo
    X_Train_All{i}=(X_Train_All{i}-ones(size(X_Train_All{i},1),1)*mu)*Gw';
    X_Test_All{i}=(X_Test_All{i}-ones(size(X_Test_All{i},1),1)*mu)*Gw';
end

disp('Whitening Matrix: ')
disp(Gw')
for j=1:ClassNo-1
    [ G{j} , E(j) ] = LDA( X_Train_All , ClassNo , j );
end
disp('Fisher Critria=')
disp(E)
d=input('Please Enter Reducted Dimention:   ');

%% LDA 
if d<=ClassNo-1
    

    %%
    disp('LDA Projection Matrix=')
    disp(G{d}')

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
