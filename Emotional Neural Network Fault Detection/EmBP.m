clc;clear; close all;
load data_train.mat
load data_test.mat

X_train = {  X_train_n , X_train_f16 , X_train_f17 , X_train_f18 } ;
X_test = { X_test_n , X_test_f16 , X_test_f17 , X_test_f18 } ;

ClassNo = size( X_train , 2 ) ;

%%

x_test = [] ;
x_train = [] ;
for i=1:ClassNo
    x_train=[x_train; X_train{i}];
    x_test=[x_test; X_test{i}];
    [n_train(i) ~]=size(X_train{i});
    [n_test(i) ~]=size(X_test{i});
end

Min=min(x_train);
h=max(x_train)-min(x_train);
clear X_train X_test 
%%
X_train=((x_train-ones(size(x_train,1),1)*Min)./(ones(size(x_train,1),1)*h))*2-1;
X_test=((x_test-ones(size(x_test,1),1)*Min)./(ones(size(x_test,1),1)*h))*2-1;

T=[];
for i=1:ClassNo
    temp=0.3*rand(n_train(i),ClassNo);
    temp(:,i)=0.7+0.3*rand(n_train(i),1);
    T=[T ;temp];
end

%%  
[ ~ , in ] = size(X_train) ; 
    
l = 5 ;
alfa = 0.1;
eta = 0.2;
X_b = 1 ;

%% Hidden Layer
W_hi = rand(in,l)*0.3 ;
W_hb = rand(1,l)*0.3 ;
W_hm = rand(1,l)*0.3 ;

%% Output Layer
W_jh = rand(l,ClassNo)*0.3 ;
W_jb = rand(1,ClassNo)*0.3 ;
W_jm = rand(1,ClassNo)*0.3 ;

%%
W_hi_t_1 = zeros(in,l) ;
W_hb_t_1 = zeros(1,l) ;
W_hm_t_1 = zeros(1,l) ;
W_jh_t_1 = zeros(l,ClassNo) ;
W_jb_t_1 = zeros(1,ClassNo) ;
W_jm_t_1 = zeros(1,ClassNo) ;

%%   Pn=S^2*XI
pn=sum(n_train);
Y_PAT = 1/in * sum( X_train , 2 );

for t=1:25
    if t==1
        k=0;
        mu=1;
    else
        k=K(t-1);
        mu=Mu(t-1);
    end
    
    for p=1:pn
        XI= X_train(p,:) ;
        YI= XI ;

        TP_hc= YI * W_hi ;
        TP_hb = X_b * W_hb ;
        TP_hm = Y_PAT(p) * W_hm ;
        X_Hh = TP_hc + TP_hb + TP_hm ;


        Y_H = 1./(1 + exp( - X_Hh ));
        TP_jc = Y_H * W_jh ;

        TP_jb = X_b * W_jb ;
        TP_jm = Y_PAT(p) * W_jm ;

        XJ = TP_jc + TP_jb + TP_jm ;

        YJ = 1./( 1 + exp( - XJ ) ) ;

        e(p) = sum( ( T(p,:) - YJ ) .^2 ) ;

        Y_AvPAT = 1/p * sum( Y_PAT(1:p) ) ;

        W_hi_t = W_hi ;
        W_hb_t = W_hb ;
        W_hm_t = W_hm ;
        W_jh_t = W_jh ;
        W_jb_t = W_jb ;
        W_jm_t = W_jm ;

        delta = YJ .* ( ones(size(YJ)) - YJ ) .* ( T(p,:) - YJ ) ;
        
        W_jh = W_jh + ...
            eta *  Y_H' * delta + alfa * ( W_jh - W_jh_t_1 ) + ...
            mu * Y_AvPAT * ones(size(Y_H')) * delta + k * ( W_jh - W_jh_t_1 );

        W_jb = W_jb + eta  * delta + alfa * ( W_jb - W_jb_t_1 );

        W_jm = W_jm + mu *  Y_AvPAT * delta + k * ( W_jm - W_jm_t_1 );

        delta_h = Y_H .* ( ones(size(Y_H)) - Y_H ) .* (  delta * W_jh' ) ;

        W_hi = W_hi +...
            eta *  YI' * delta_h + alfa * ( W_hi - W_hi_t_1 ) +...
            mu *  Y_AvPAT * ones(size(YI')) * delta_h + k * ( W_hi - W_hi_t_1 );

        W_hb = W_hb + eta * delta_h + alfa * ( W_hb - W_hb_t_1 );

        W_hm = W_hm + mu *  Y_AvPAT * delta_h + k * ( W_hm - W_hm_t_1 );

        W_hi_t_1 = W_hi ;
        W_hb_t_1 = W_hb ;
        W_hm_t_1 = W_hm ;
        W_jh_t_1 = W_jh ;
        W_jb_t_1 = W_jb ;
        W_jm_t_1 = W_jm ;
        
    end
        
    E(t) = sum( e )/ ( pn * ClassNo )  ;

    Mu(t) = Y_AvPAT + E(t) ;
    if t==1
        Mu0=Mu;
    end
    K(t) = Mu0 - Mu(t) ;
    
end


%%
figure
plot(Mu)
hold all
plot(K)
hold off
grid on
legend('Anxiety Coefficient','Confidence Coefficient');
title('Anxiety & Confidence Coefficients VS. Iteration')
xlabel('Iteraion')
ylabel('Anxiety & Confidence Coefficients')

%% Test Part
Y_PAT = 1/in * sum( X_test , 2 );
pn=sum(n_test);
for p=1:pn
    XI= X_test(p,:) ;
    YI= XI ;

    TP_hc= YI * W_hi ;
    TP_hb = X_b * W_hb ;
    TP_hm = Y_PAT(p) * W_hm ;
    X_Hh = TP_hc + TP_hb + TP_hm ;


    Y_H = 1./(1 + exp( - X_Hh ));
    TP_jc = Y_H * W_jh ;

    TP_jb = X_b * W_jb ;
    TP_jm = Y_PAT(p) * W_jm ;

    XJ = TP_jc + TP_jb + TP_jm ;

    YJ(p,:) = 1./( 1 + exp( - XJ ) ) ;
    tem=YJ(p,:);
    Class(p)=ClassNo+1;
    f=1;
    rc=0;
    while ( rc==0 && f <= ClassNo)
        if tem(f)>=0.7
            tem(f)=[];
            rc=1;
            if tem<=0.3
                Class(p)=f;
            end
        end
        f=f+1;
    end
end
figure
plot(YJ)
grid on
title('Test Output VS. Sample')
xlabel('Sample')
ylabel('Test Output')

for n=1:ClassNo
    C=Class(1:n_test(n));
    Class(1:n_test(n))=[];
    for j=1:ClassNo+1
       Confidence(n,j)=size(C(C==j),2)/n_test(n);
    end
end
Confidence