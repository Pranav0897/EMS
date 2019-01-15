function [P,Q,Pij,Qij] = measurements(linedatas,V,phi,ybus)
%% power injection measurements
nbus = max(max(linedatas(:,1)),max(linedatas(:,2)));
nbranch=length(linedatas(:,1));
b = linedatas(:,5);
fb = linedatas(:,1);
tb = linedatas(:,2);
G = real(ybus);
B = imag(ybus);
P = zeros(nbus,1);
Q = zeros(nbus,1);
for l = 1:nbus
        for m = 1:nbus
            P(l) = P(l) + V(l)* V(m)*(G(l,m)*cos(phi(l)-phi(m)) + B(l,m)*sin(phi(l)-phi(m)));
            Q(l) = Q(l) + V(l)* V(m)*(G(l,m)*sin(phi(l)-phi(m)) - B(l,m)*cos(phi(l)-phi(m))); 
        end
end
    bbus_perturbed = zeros(nbus,nbus);
    for k=1:nbranch
        bbus_perturbed(fb(k),tb(k)) = b(k) ./ 2;
        bbus_perturbed(tb(k),fb(k)) = bbus_perturbed(fb(k),tb(k));
    end
%% power flows calculation
    for  o = 1:nbranch
        m = fb(o);
        n = tb(o);
        Pij(o,1) = -V(m)^2*(G(m,n)) + V(m)*V(n)*(G(m,n)*cos(phi(m)-phi(n))+ B(m,n)*sin(phi(m)-phi(n)));
        Qij(o,1) = V(m)^2*(B(m,n)- bbus_perturbed(m,n)) +V(m)*V(n)*(G(m,n)*sin(phi(m)-phi(n)) - B(m,n)*cos(phi(m)-phi(n)));
        Pji(o,1) = -V(n)^2*(G(n,m)) + V(n)*V(m)*(G(n,m)*cos(phi(n)-phi(m))+ B(n,m)*sin(phi(n)-phi(m)));
        Qji(o,1) = V(n)^2*(B(n,m)- bbus_perturbed(n,m)) +V(n)*V(m)*(G(n,m)*sin(phi(n)-phi(m)) - B(n,m)*cos(phi(n)-phi(m)));
    end