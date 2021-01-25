function [Class_Out]=gaussianpdf_risk_classifier(Xtest,mu,sigma,Landa)
    d=size(sigma,1);
    p=zeros(1,2);
    for i=1:2
        z=Xtest-mu(:,:,i);
        p(i)=1/((2*pi)^(d/2)*det(sigma(:,:,i))^0.5)*exp(-1/2*z*(sigma(:,:,i)^-1)*z');
    end
    if (p(1)/p(2))>((Landa(2,1)-Landa(2,2))/(Landa(1,2)-Landa(1,1)))
        Class_Out=1;
    elseif (p(1)/p(2))<((Landa(2,1)-Landa(2,2))/(Landa(1,2)-Landa(1,1)))
        Class_Out=2;
    else
        Class_Out=3;
    end

end