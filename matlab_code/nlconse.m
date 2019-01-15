function [c ceq]=nlconse(h,P,Q,Pij,Qij,linedatas,ybus)
nbus = max(max(linedatas(:,1)),max(linedatas(:,2)));
nbranch=length(linedatas(:,1));
Vse = h(length(P)+length(Q)+length(Pij)+length(Qij)+1:length(P)+length(Q)+length(Pij)+length(Qij)+nbus);
phise = h(length(P)+length(Q)+length(Pij)+length(Qij)+nbus+1:length(P)+length(Q)+length(Pij)+length(Qij)+2*nbus);
nbus = max(max(linedatas(:,1)),max(linedatas(:,2)));
nbranch=length(linedatas(:,1));
b = linedatas(:,5);
fb = linedatas(:,1);
tb = linedatas(:,2);
G = real(ybus);
B = imag(ybus);
Pcal = zeros(nbus,1);
Qcal = zeros(nbus,1);
for l = 1:nbus
        for m = 1:nbus
            Pcal(l,1) = Pcal(l) + Vse(l)* Vse(m)*(G(l,m)*cos(phise(l)-phise(m)) + B(l,m)*sin(phise(l)-phise(m)));
            Qcal(l,1) = Qcal(l) + Vse(l)* Vse(m)*(G(l,m)*sin(phise(l)-phise(m)) - B(l,m)*cos(phise(l)-phise(m))); 
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
        Pijcal(o,1) = -Vse(m)^2*(G(m,n)) + Vse(m)*Vse(n)*(G(m,n)*cos(phise(m)-phise(n))+ B(m,n)*sin(phise(m)-phise(n)));
        Qijcal(o,1) = Vse(m)^2*(B(m,n)- bbus_perturbed(m,n)) +Vse(m)*Vse(n)*(G(m,n)*sin(phise(m)-phise(n)) - B(m,n)*cos(phise(m)-phise(n)));
        
    end
    ceq = [Pcal-P-h(1:length(P));Qcal-Q-h(length(P)+1:length(P)+length(Q));Pijcal-Pij-h(length(P)+length(Q)+1:length(P)+length(Q)+length(Pij));Qijcal-Qij-h(length(P)+length(Q)+length(Pij)+1:length(P)+length(Q)+length(Pij)+length(Qij))]
    c =[];