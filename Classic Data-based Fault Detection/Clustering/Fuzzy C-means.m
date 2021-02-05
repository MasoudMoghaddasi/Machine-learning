clc;clear;close all
load data_clustering.mat

X_Train=data1;

c=input('Please Input The Number Of The Clusters:    ');
[center,U,obj_fcn] = fcm(X_Train, c);
plot(X_Train(:,1), X_Train(:,2),'o');
grid on
maxU = max(U);
hold on
for i=1:c
    switch i
        case 1
            co='r*';
        case 2
            co='b*';
        case 3
            co='g*';
        case 4
            co='m*';
    end
            
plot(X_Train(U(i,:) == maxU,1),X_Train(U(i,:) == maxU,2),co)
hold on
end
 plot(center(:,1),center(:,2),'kx',...
     'MarkerSize',12,'LineWidth',2)
plot(center(:,1),center(:,2),'ko',...
     'MarkerSize',12,'LineWidth',2)
hold off
switch i
    case 1
        legend('All Datas','Cluster 1','Centroids',...
       'Location','NW')
    case 2
        legend('All Datas','Cluster 1','Cluster 2','Centroids',...
       'Location','NW')
    case 3
        legend('All Datas','Cluster 1','Cluster 2','Cluster 3','Centroids',...
       'Location','NW')
    case 4
        legend('All Datas','Cluster 1','Cluster 2','Cluster 3','Cluster 4','Centroids',...
       'Location','NW')
end


figure
X_Train=data2;


[center,U,obj_fcn] = fcm(X_Train, c);
plot(X_Train(:,1), X_Train(:,2),'o');
grid on
maxU = max(U);
hold on
for i=1:c
    switch i
        case 1
            co='r*';
        case 2
            co='b*';
        case 3
            co='g*';
        case 4
            co='m*';
    end
            
plot(X_Train(U(i,:) == maxU,1),X_Train(U(i,:) == maxU,2),co)
hold on
end
 plot(center(:,1),center(:,2),'kx',...
     'MarkerSize',12,'LineWidth',2)
plot(center(:,1),center(:,2),'ko',...
     'MarkerSize',12,'LineWidth',2)
hold off
switch i
    case 1
        legend('All Datas','Cluster 1','Centroids',...
       'Location','NW')
    case 2
        legend('All Datas','Cluster 1','Cluster 2','Centroids',...
       'Location','NW')
    case 3
        legend('All Datas','Cluster 1','Cluster 2','Cluster 3','Centroids',...
       'Location','NW')
    case 4
        legend('All Datas','Cluster 1','Cluster 2','Cluster 3','Cluster 4','Centroids',...
       'Location','NW')
end
