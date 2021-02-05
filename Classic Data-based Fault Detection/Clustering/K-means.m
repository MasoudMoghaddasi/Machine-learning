clc;clear;close all
load data_clustering.mat

X_Train=data1;

c=input('Please Input The Number Of The Clusters:    ');

figure
[idx,ctrs] = kmeans(X_Train, c);
for i=1:c
    switch i
        case 1
            co='r.';
        case 2
            co='b.';
        case 3
            co='g.';
        case 4
            co='m.';
    end
            
plot(X_Train(idx==i,1),X_Train(idx==i,2),co,'MarkerSize',12)
hold on
end

plot(ctrs(:,1),ctrs(:,2),'kx',...
     'MarkerSize',12,'LineWidth',2)
plot(ctrs(:,1),ctrs(:,2),'ko',...
     'MarkerSize',12,'LineWidth',2)
 
switch i
    case 1
        legend('Cluster 1','Centroids',...
       'Location','NW')
    case 2
        legend('Cluster 1','Cluster 2','Centroids',...
       'Location','NW')
    case 3
        legend('Cluster 1','Cluster 2','Cluster 3','Centroids',...
       'Location','NW')
    case 4
        legend('Cluster 1','Cluster 2','Cluster 3','Cluster 4','Centroids',...
       'Location','NW')
end

grid on



X_Train=data2;

figure
[idx,ctrs] = kmeans(X_Train, c);
for i=1:c
    switch i
        case 1
            co='r.';
        case 2
            co='b.';
        case 3
            co='g.';
        case 4
            co='m.';
    end
            
plot(X_Train(idx==i,1),X_Train(idx==i,2),co,'MarkerSize',12)
hold on
end

plot(ctrs(:,1),ctrs(:,2),'kx',...
     'MarkerSize',12,'LineWidth',2)
plot(ctrs(:,1),ctrs(:,2),'ko',...
     'MarkerSize',12,'LineWidth',2)
 
 switch i
    case 1
        legend('Cluster 1','Centroids',...
       'Location','NW')
    case 2
        legend('Cluster 1','Cluster 2','Centroids',...
       'Location','NW')
    case 3
        legend('Cluster 1','Cluster 2','Cluster 3','Centroids',...
       'Location','NW')
    case 4
        legend('Cluster 1','Cluster 2','Cluster 3','Cluster 4','Centroids',...
       'Location','NW')
end

grid on
