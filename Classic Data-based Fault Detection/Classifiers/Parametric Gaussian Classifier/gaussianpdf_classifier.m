function [Class_Out]=gaussianpdf_classifier(Xtest,mu,sigma)
    d=size(sigma,1);
    p=zeros(1,10);
    for i=1:10
        z=Xtest-mu(:,:,i);
        p(i)=1/((2*pi)^(d/2)*det(sigma(:,:,i))^0.5)*exp(-1/2*z*(sigma(:,:,i)^-1)*z');
    end
    [MAX,Class_Out]=max(p);
    p(Class_Out)=[];
    if p~=MAX;
    else
        Class_Out=11;
    end

end